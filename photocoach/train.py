"""Train the twin judge on cached features.

    python -m photocoach.train            # uses whatever is in data/features/

Supervision (all real, no formula-made pseudo labels):
  AVA    ~140k photo-contest images, 78-549 human votes each  -> aesthetic
  AADB   10k Flickr photos rated by 5 people on 11 attributes -> composition, light, colour, DoF ...
  SPAQ   11k smartphone photos, MOS + 5 attribute ratings      -> technical, exposure, noise, sharpness ...
  KonIQ  10k in-the-wild photos, ~120 ratings each             -> technical
  synth  clean original vs single known fault                  -> exposure, WB, level, framing ...

Every batch is trained the twin way: pairs of photos go through the shared
tower and the loss is on which one people preferred (plus a small regression
term that anchors the absolute scale so scores are meaningful for one photo).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from .model import CKPT, HEAD_NAMES, HIDX, K, TwinJudge
from .synth import AESTHETIC_EFFECTS, EFFECT_HEAD
from .backbone import L_DIM, N_LOCAL

FEAT = os.path.join(os.path.dirname(CKPT), "..", "data", "features")

# dataset -> {label column: head}
LABEL_MAP = {
    "ava":   {"mos": "aesthetic"},
    "aadb":  {"score": "aesthetic", "BalacingElements": "balance", "RuleOfThirds": "thirds", "Object": "subject_emphasis",
              "Symmetry": "symmetry", "Repetition": "repetition", "Light": "lighting", "ColorHarmony": "color_harmony",
              "VividColor": "vivid_color", "DoF": "depth_of_field", "MotionBlur": "motion_blur", "Content": "content"},
    "spaq":  {"MOS": "technical", "Brightness": "exposure", "Contrast": "contrast", "Colorfulness": "vivid_color",
              "Sharpness": "sharpness", "Noisiness": "noise"},
    "koniq": {"mos": "technical"},
    "pros":  {},
}
SAMPLE_P = {"ava": 0.34, "aadb": 0.22, "spaq": 0.16, "koniq": 0.08, "synth": 0.20}
SYNTH_ONLY = ["framing", "level", "color_accuracy"]


def load(dev):
    D, synth_pairs = {}, {"train": [], "test": []}
    offset_note = []
    for name in LABEL_MAP:
        p = os.path.join(FEAT, f"{name}.npz")
        if not os.path.exists(p):
            print(f"  (no features for {name} yet - skipping)")
            continue
        z = np.load(p)
        n = len(z["key"])
        g = torch.tensor(z["g"].astype(np.float32))
        has_l = "l" in z.files
        l = torch.tensor(z["l"].astype(np.float32)) if has_l else torch.zeros(n, N_LOCAL, L_DIM)
        h = torch.tensor(z["h"])
        split, eff, sev, key = z["split"], z["effect"], z["severity"], z["key"]
        orig = eff == ""
        Y = np.full((n, K), np.nan, np.float32)
        tr = orig & (split != "test")
        for col, head in LABEL_MAP[name].items():
            y = z["y_" + col].astype(np.float32)
            mu, sd = np.nanmean(y[tr]), np.nanstd(y[tr]) + 1e-6
            Y[orig, HIDX[head]] = ((y - mu) / sd)[orig]
        d = dict(g=g.to(dev), l=l.to(dev), h=h.to(dev), has_l=torch.full((n,), float(has_l), device=dev),
                 Y=torch.tensor(Y).to(dev), orig=orig, split=split, key=key, n=n)
        d["train_idx"] = torch.tensor(np.where(orig & (split != "test"))[0], device=dev)
        d["test_idx"] = np.where(orig & (split == "test"))[0]
        # AVA: keep the vote spread so pair targets can be "share of raters preferring A"
        if name == "ava":
            d["std"] = torch.tensor(z["y_std"]).to(dev)
            d["raw"] = torch.tensor(z["y_mos"]).to(dev)
        D[name] = d
        # synthetic (original, variant) pairs inside this file
        first = {}
        for i in np.where(orig)[0]:
            first[key[i]] = i
        for i in np.where(~orig)[0]:
            o = first.get(key[i])
            if o is not None:
                synth_pairs["test" if split[i] == "test" else "train"].append((name, int(o), int(i), str(eff[i]), float(sev[i])))
        print(f"  {name}: {n} records ({int(orig.sum())} originals, {int((~orig).sum())} synthetic variants, local={has_l})")
    return D, synth_pairs


def feats(d, idx):
    return d["g"][idx], d["l"][idx], d["h"][idx], d["has_l"][idx]


def pair_loss(model, s, Y, extra=None):
    """Twin ranking + anchor regression on one dataset batch.  s,Y: (B,K)."""
    B = s.shape[0]
    perm = torch.randperm(B, device=s.device)
    mask = ~torch.isnan(Y)
    Yz = torch.nan_to_num(Y)
    logits = model.compare_logits(s, s[perm])
    if extra is not None:   # AVA: probability a random rater scores i above j
        mu, sd = extra
        tgt_a = torch.distributions.Normal(0, 1).cdf((mu - mu[perm]) / torch.sqrt(sd ** 2 + sd[perm] ** 2 + 1e-6))
        tgt = torch.sigmoid((Yz - Yz[perm]) / 0.6)
        tgt[:, HIDX["aesthetic"]] = tgt_a
    else:
        tgt = torch.sigmoid((Yz - Yz[perm]) / 0.6)
    m = (mask & mask[perm]).float()
    rank = (F.binary_cross_entropy_with_logits(logits, tgt, reduction="none") * m).sum() / m.sum().clamp(min=1)
    reg = (F.smooth_l1_loss(s, Yz, reduction="none") * mask.float()).sum() / mask.float().sum().clamp(min=1)
    return rank + 0.5 * reg


def synth_loss(model, D, pairs, B, dev, rng):
    pick = rng.integers(0, len(pairs), B)
    by_ds = {}
    for j in pick:
        by_ds.setdefault(pairs[j][0], []).append(pairs[j])
    loss, cnt = 0.0, 0
    for name, ps in by_ds.items():
        d = D[name]
        o = torch.tensor([p[1] for p in ps], device=dev)
        v = torch.tensor([p[2] for p in ps], device=dev)
        sev = torch.tensor([p[4] for p in ps], device=dev)
        so, sv = model.score(*feats(d, o)), model.score(*feats(d, v))
        logits = model.compare_logits(so, sv)
        hk = torch.tensor([HIDX[EFFECT_HEAD[p[3]]] if EFFECT_HEAD[p[3]] else -1 for p in ps], device=dev)
        ok = torch.tensor([HIDX["aesthetic"] if p[3] in AESTHETIC_EFFECTS else HIDX["technical"] for p in ps], device=dev)
        tgt = 0.70 + 0.28 * sev                               # mild fault = mild preference
        ar = torch.arange(len(ps), device=dev)
        l_over = F.binary_cross_entropy_with_logits(logits[ar, ok], tgt, reduction="none")
        has = hk >= 0
        l_attr = F.binary_cross_entropy_with_logits(logits[ar, hk.clamp(min=0)], tgt, reduction="none") * has
        # anchor the synthetic-only heads on an absolute scale (clean ~ +0.7, faulty lower with severity)
        syn_only = torch.tensor([EFFECT_HEAD[p[3]] in SYNTH_ONLY for p in ps], device=dev)
        hk0 = hk.clamp(min=0)
        anch = (F.smooth_l1_loss(sv[ar, hk0], 0.7 - 3.0 * sev, reduction="none") * syn_only).sum() / syn_only.sum().clamp(min=1)
        for nm in SYNTH_ONLY:   # every clean original should sit high on these
            anch = anch + 0.3 * F.smooth_l1_loss(so[:, HIDX[nm]], torch.full_like(sev, 0.7))
        # an unrelated fault must NOT move another synthetic-only head (keeps heads specific)
        other = torch.ones(len(ps), K, device=dev, dtype=torch.bool)
        other[ar, hk0] = False
        keep = torch.zeros(K, device=dev, dtype=torch.bool)
        keep[[HIDX[nm] for nm in SYNTH_ONLY]] = True
        spec = (((sv - so.detach()) ** 2) * (other & keep).float()).mean()
        # ...except that a tilt/crop changes framing/level legitimately only for its own head
        loss = loss + (l_over.mean() + l_attr.sum() / has.sum().clamp(min=1) + 0.5 * anch + 0.5 * spec) * len(ps)
        cnt += len(ps)
    return loss / max(cnt, 1)


@torch.no_grad()
def evaluate(model, D, synth_pairs, dev):
    model.eval()
    rep = {}
    for name, d in D.items():
        idx = d["test_idx"]
        if len(idx) == 0:
            continue
        ti = torch.tensor(idx, device=dev)
        s = model.score(*feats(d, ti)).cpu().numpy()
        Y = d["Y"][ti].cpu().numpy()
        for k in range(K):
            ok = ~np.isnan(Y[:, k])
            if ok.sum() > 30:
                rep[f"{name}/{HEAD_NAMES[k]}"] = dict(srcc=float(spearmanr(s[ok, k], Y[ok, k]).correlation), n=int(ok.sum()))
                # twin accuracy: random pairs whose human scores clearly differ
                rng = np.random.default_rng(0)
                a, b = rng.integers(0, ok.sum(), 20000), rng.integers(0, ok.sum(), 20000)
                ya, yb = Y[ok, k][a], Y[ok, k][b]
                clear = np.abs(ya - yb) > 1.0
                if clear.sum() > 50:
                    rep[f"{name}/{HEAD_NAMES[k]}"]["pair_acc_clear"] = float(((s[ok, k][a] > s[ok, k][b]) == (ya > yb))[clear].mean())
                anyp = np.abs(ya - yb) > 1e-6
                rep[f"{name}/{HEAD_NAMES[k]}"]["pair_acc_all"] = float(((s[ok, k][a] > s[ok, k][b]) == (ya > yb))[anyp].mean())
    # synthetic: does the clean original win, on the right attribute?
    acc = {}
    for name, o, v, eff, sev in synth_pairs["test"]:
        d = D[name]
        so = model.score(*feats(d, torch.tensor([o], device=dev)))[0]
        sv = model.score(*feats(d, torch.tensor([v], device=dev)))[0]
        ok = HIDX["aesthetic"] if eff in AESTHETIC_EFFECTS else HIDX["technical"]
        acc.setdefault(eff, []).append((float(so[ok] > sv[ok]), float(so[HIDX[EFFECT_HEAD[eff]]] > sv[HIDX[EFFECT_HEAD[eff]]]) if EFFECT_HEAD[eff] else np.nan))
    rep["synth"] = {e: dict(overall_acc=float(np.mean([x[0] for x in v])), attr_acc=float(np.nanmean([x[1] for x in v])) if EFFECT_HEAD[e] else None, n=len(v)) for e, v in sorted(acc.items())}
    model.train()
    return rep


def main(steps=9000, B=384, lr=6e-4, seed=0):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    D, synth_pairs = load(dev)
    print(f"  synthetic pairs: {len(synth_pairs['train'])} train / {len(synth_pairs['test'])} test")
    model = TwinJudge().to(dev)
    # standardise measured signals on the pooled training data
    H = torch.cat([d["h"][d["train_idx"]] for d in D.values()])
    med = H.median(0).values
    mad = (H - med).abs().median(0).values * 1.4826 + 1e-4
    model.tower.h_mean.copy_(med); model.tower.h_std.copy_(mad)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.06)
    names = [n for n in SAMPLE_P if (n in D and len(LABEL_MAP.get(n, {})) > 0) or (n == "synth" and synth_pairs["train"])]
    probs = np.array([SAMPLE_P[n] for n in names]); probs /= probs.sum()
    t0 = time.time()
    for step in range(steps):
        name = names[rng.choice(len(names), p=probs)]
        if name == "synth":
            loss = synth_loss(model, D, synth_pairs["train"], B, dev, rng)
        else:
            d = D[name]
            idx = d["train_idx"][torch.randint(0, len(d["train_idx"]), (B,), device=dev)]
            g, l, h, has_l = feats(d, idx)
            if d["has_l"][0] > 0:   # local-view dropout so the tower also works from the global view alone
                has_l = has_l * (torch.rand(B, device=dev) > 0.15).float()
            s = model.score(g, l, h, has_l)
            extra = (d["raw"][idx], d["std"][idx]) if name == "ava" else None
            loss = pair_loss(model, s, d["Y"][idx], extra)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step(); sched.step()
        if step % 1000 == 0 or step == steps - 1:
            print(f"step {step:5d}  loss {loss.item():.4f}  ({time.time() - t0:.0f}s)", flush=True)

    rep = evaluate(model, D, synth_pairs, dev)
    model.eval()

    # ---- calibration: where does a score sit among real photos? ----
    with torch.no_grad():
        pool = []
        for name in ("aadb", "koniq", "spaq", "pros"):
            if name in D:
                d = D[name]
                idx = torch.tensor(np.where(d["orig"])[0], device=dev)
                pool.append(model.score(*feats(d, idx)).cpu())
        pool = torch.cat(pool)
        model.q_table.copy_(torch.quantile(pool, torch.linspace(0, 1, 101), dim=0).T.contiguous())
        model.ref_mean.copy_(pool.mean(0)); model.ref_std.copy_(pool.std(0))

    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    meta = dict(report=rep, heads=HEAD_NAMES, datasets={n: int(d["n"]) for n, d in D.items()},
                pool_size=int(len(pool)), steps=steps, trained=time.strftime("%Y-%m-%d %H:%M"))
    torch.save(dict(state=model.state_dict(), meta=meta), CKPT)
    json.dump(meta, open(os.path.join(os.path.dirname(CKPT), "report.json"), "w"), indent=1)

    print("\n=== held-out results (test images never seen in training) ===")
    for k, v in rep.items():
        if k != "synth":
            print(f"  {k:28s} SRCC {v['srcc']:.3f}   pair-acc(clear) {v.get('pair_acc_clear', float('nan')):.3f}   pair-acc(all) {v['pair_acc_all']:.3f}   n={v['n']}")
    print("  synthetic clean-vs-fault pairs (overall acc / right-attribute acc):")
    for e, v in rep["synth"].items():
        print(f"    {e:16s} {v['overall_acc']:.3f} / {v['attr_acc'] if v['attr_acc'] is None else round(v['attr_acc'], 3)}   n={v['n']}")
    print(f"saved -> {CKPT}")


if __name__ == "__main__":
    kw = {a.split("=")[0].lstrip("-"): int(a.split("=")[1]) for a in sys.argv[1:] if "=" in a}
    main(**kw)
