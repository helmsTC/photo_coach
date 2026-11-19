"""Cache backbone features + measured signals for every training image.

    python -m photocoach.extract aadb koniq spaq pros ava

Images are read straight out of the downloaded archives (nothing is unpacked).
Output: data/features/<name>.npz
"""
from __future__ import annotations

import csv
import io
import os
import sys
import tarfile
import time
import zipfile
import zlib
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import torch

from . import signals, synth
from .backbone import Backbone, global_view, load_image, local_views

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
FEAT = os.path.join(ROOT, "data", "features")

AADB_ATTRS = ["score", "BalacingElements", "ColorHarmony", "Content", "DoF", "Light", "MotionBlur",
              "Object", "Repetition", "RuleOfThirds", "Symmetry", "VividColor"]


# ----------------------------------------------------------------------------
# sources: each yields (key, bytes, meta) ; meta = dict(labels..., split=..., n_synth=int, local=bool)
# ----------------------------------------------------------------------------
def src_aadb():
    lab = defaultdict(dict)
    split = {}
    for sp, tag in (("train", "Train"), ("val", "Validation"), ("test", "Test"), ("test", "TestNew")):
        for a in AADB_ATTRS:
            p = os.path.join(RAW, "aadb_labels", f"AADB__imgListFiles_label__imgList{tag}Regression_{a}.txt")
            for line in open(p):
                parts = line.split()
                if len(parts) == 2:
                    lab[parts[0]][a] = float(parts[1]); split[parts[0]] = sp
    d = os.path.join(RAW, "AADB_aesthetic_IQA_eccv16")
    for zn in ("datasetImages_originalSize.zip", "AADB_newtest_originalSize.zip"):
        zf = zipfile.ZipFile(os.path.join(d, zn))
        for n in zf.namelist():
            b = os.path.basename(n)
            if b in lab and len(lab[b]) == len(AADB_ATTRS):
                yield b, zf.read(n), dict(lab[b], split=split[b], local=True, n_synth=0)
                lab.pop(b)


def src_koniq():
    rows = {r["img_name"]: r for r in csv.DictReader(open(os.path.join(RAW, "meta_info_KonIQ10kDataset.csv")))}
    thr = np.percentile([float(r["mos"]) for r in rows.values()], 88)
    zf = zipfile.ZipFile(os.path.join(RAW, "koniq10k.zip"))
    for n in zf.namelist():
        b = os.path.basename(n)
        if b in rows:
            r = rows.pop(b)
            mos = float(r["mos"])
            sp = {"train": "train", "val": "val", "test": "test"}.get(r["official_split"], "train")
            yield b, zf.read(n), dict(mos=mos, split=sp, local=True, n_synth=2 if (mos >= thr and sp == "train") else 0)


def _tar_stream(path, wanted):
    with tarfile.open(path, "r|gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            b = os.path.basename(m.name)
            if b in wanted:
                yield b, tf.extractfile(m).read()


def src_spaq():
    cols = ["MOS", "Brightness", "Colorfulness", "Contrast", "Noisiness", "Sharpness"]
    rows = {r["Image name"]: r for r in csv.DictReader(open(os.path.join(RAW, "meta_info_SPAQDataset.csv")))}
    thr = np.percentile([float(r["MOS"]) for r in rows.values()], 88)
    for b, data in _tar_stream(os.path.join(RAW, "spaq.tgz"), rows):
        r = rows[b]
        sp = r["ratio802_seed123_split_01"]
        sp = "test" if sp == "val" else "train"   # pyiqa 80/20 split
        meta = {c: float(r[c]) for c in cols}
        meta.update(split=sp, local=True, n_synth=2 if (meta["MOS"] >= thr and sp == "train") else 0)
        yield b, data, meta


def src_ava(n_train=70000, n_test=8000, seed=0):
    rows = list(csv.DictReader(open(os.path.join(RAW, "meta_info_AVADataset.csv"))))
    rng = np.random.default_rng(seed)
    is_test = np.array([r["official_split"] != "0" for r in rows])
    # official_split: 0 = train, others = val/test
    idx_tr = rng.permutation(np.where(~is_test)[0])[:n_train]
    idx_te = rng.permutation(np.where(is_test)[0])[:n_test]
    wanted = {}
    for i in idx_tr: wanted[rows[i]["img_name"]] = (rows[i], "train")
    for i in idx_te: wanted[rows[i]["img_name"]] = (rows[i], "test")
    for b, data in _tar_stream(os.path.join(RAW, "ava.tgz"), wanted):
        r, sp = wanted[b]
        votes = np.array([float(r[f"c{k}"]) for k in range(1, 11)])
        mean = float((votes * np.arange(1, 11)).sum() / max(votes.sum(), 1))
        std = float(np.sqrt((votes * (np.arange(1, 11) - mean) ** 2).sum() / max(votes.sum(), 1)))
        yield b, data, dict(mos=mean, std=std, split=sp, local=False, n_synth=0)


def src_pros():
    d = os.path.join(ROOT, "dataset", "professional_images")
    names = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    seen = set()
    for i, f in enumerate(names):
        data = open(os.path.join(d, f), "rb").read()
        hsh = zlib.crc32(data[:200000])
        if hsh in seen:       # the folder holds duplicate downloads
            continue
        seen.add(hsh)
        yield f, data, dict(split="test" if i % 7 == 0 else "train", local=True, n_synth=4)


SOURCES = {"aadb": src_aadb, "koniq": src_koniq, "spaq": src_spaq, "ava": src_ava, "pros": src_pros}


# ----------------------------------------------------------------------------
def _init_worker():
    import cv2
    cv2.setNumThreads(1)


def _prep(args):
    key, data, meta = args
    try:
        rgb = signals.to_std(np.asarray(load_image(data, max_side=1536)))
        if min(rgb.shape[:2]) < 48:
            return None
        variants = [("", 0.0, rgb)]
        if meta["n_synth"]:
            rng = np.random.default_rng(zlib.crc32(key.encode()))
            variants += synth.random_variants(rgb, meta["n_synth"], rng)
        out = []
        for eff, sev, im in variants:
            out.append(dict(key=key, effect=eff, severity=sev, meta=meta,
                            gv=global_view(im),
                            lv=local_views(im) if meta["local"] else None,
                            h=signals.vector(signals.measure(im, full=False))))
        return out
    except Exception as e:  # corrupt image etc.
        print(f"  skip {key}: {e}", flush=True)
        return None


def run(name: str, bb: Backbone, limit: int | None = None):
    os.makedirs(FEAT, exist_ok=True)
    out_path = os.path.join(FEAT, f"{name}.npz")
    recs = []                       # per-record dicts (without pixel data)
    G, L = {}, {}
    buckets = defaultdict(list)     # global-view shape -> [(rec_index, gv)]
    lbuf = []                       # [(rec_index, lv)]

    def flush_g(shape):
        items = buckets.pop(shape, [])
        if items:
            emb = bb.embed_global(np.stack([x[1] for x in items]))
            for (i, _), e in zip(items, emb):
                G[i] = e.numpy()

    def flush_l():
        if lbuf:
            emb = bb.embed_local(np.concatenate([x[1] for x in lbuf])).view(len(lbuf), 4, -1)
            for (i, _), e in zip(lbuf, emb):
                L[i] = e.numpy()
            lbuf.clear()

    def save(final=False):
        for s in list(buckets):
            flush_g(s)
        flush_l()
        n = len(recs)
        label_keys = sorted({k for r in recs for k in r["meta"] if k not in ("split", "local", "n_synth")})
        arr = dict(
            key=np.array([r["key"] for r in recs]),
            effect=np.array([r["effect"] for r in recs]),
            severity=np.array([r["severity"] for r in recs], np.float32),
            split=np.array([r["meta"]["split"] for r in recs]),
            g=np.stack([G[i] for i in range(n)]).astype(np.float16),
            h=np.stack([r["h"] for r in recs]),
        )
        if L:
            arr["l"] = np.stack([L[i] for i in range(n)]).astype(np.float16)
        for k in label_keys:
            arr["y_" + k] = np.array([r["meta"].get(k, np.nan) for r in recs], np.float32)
        tmp = out_path + ".tmp.npz"
        np.savez(tmp, **arr)
        os.replace(tmp, out_path)
        print(f"[{name}] saved {n} records -> {out_path}{' (final)' if final else ''}", flush=True)

    t0 = time.time()
    last_save = 0
    src = SOURCES[name]()
    with ProcessPoolExecutor(14, mp_context=mp.get_context('spawn'), initializer=_init_worker) as ex:
        pending = deque()
        n_in = 0

        def drain(block_all=False):
            nonlocal last_save
            while pending and (block_all or len(pending) >= 96 or pending[0].done()):
                res = pending.popleft().result()
                for r in res or []:
                    i = len(recs)
                    gv, lv = r.pop("gv"), r.pop("lv")
                    recs.append(r)
                    buckets[gv.shape].append((i, gv))
                    if len(buckets[gv.shape]) >= 32:
                        flush_g(gv.shape)
                    if lv is not None:
                        lbuf.append((i, lv))
                        if len(lbuf) >= 24:
                            flush_l()
                if len(recs) - last_save >= 25000:
                    last_save = len(recs)
                    save()
                if len(recs) and len(recs) % 500 < len(res or []):
                    el = time.time() - t0
                    print(f"[{name}] {len(recs)} records  {len(recs) / el:.1f}/s  {el / 60:.1f} min", flush=True)

        for item in src:
            pending.append(ex.submit(_prep, item))
            n_in += 1
            drain()
            if limit and n_in >= limit:
                break
        drain(block_all=True)
    save(final=True)


if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    limit = next((int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--limit=")), None)
    bb = Backbone()
    for nm in names:
        run(nm, bb, limit)
