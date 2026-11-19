"""Head-to-head: how much accuracy does a phone-sized backbone cost?

    python -m mobile.pilot [--only=aadb,koniq] [--steps=4000] [--desktop=/path/to/desktop/features]

Trains the same twin tower on cached features from each candidate encoder and prints the
held-out numbers side by side with the desktop model (CLIP ViT-L/14 + ViT-B/16) trained on
the same datasets.  Also tries the design choices that matter for porting:
squash vs letterbox for the whole-frame view, with/without the four detail crops, and how
many hand-measured signals the tower really needs.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from photocoach import train
from photocoach.signals import VEC_KEYS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOB = os.path.join(ROOT, "data", "features_mobile")
# signals that are a few lines of arithmetic on any platform (tone, clipping, colour, size);
# the sharpness / focus / noise measurements depend on exact filter implementations
EASY = [k for k in VEC_KEYS if not k.startswith(("sharp", "edge_", "focus_", "noise_", "local_contrast"))]
SIZE = ["log_long_side", "aspect"]
AADB_ATTR = ["balance", "thirds", "subject_emphasis", "symmetry", "repetition", "lighting", "color_harmony",
             "vivid_color", "depth_of_field", "motion_blur", "content"]


def summarise(rep: dict) -> dict:
    g = lambda k: rep.get(k, {}).get("srcc", float("nan"))
    syn = rep.get("synth", {})
    return dict(
        ava=g("ava/aesthetic"), aadb=g("aadb/aesthetic"),
        attrs=float(np.nanmean([g(f"aadb/{a}") for a in AADB_ATTR])),
        koniq=g("koniq/technical"), spaq=g("spaq/technical"),
        spaq_attr=float(np.nanmean([g(f"spaq/{a}") for a in ("exposure", "contrast", "sharpness", "noise")])),
        syn=float(np.mean([v["overall_acc"] for v in syn.values()])) if syn else float("nan"),
        syn_attr=float(np.mean([v["attr_acc"] for v in syn.values() if v["attr_acc"] is not None])) if syn else float("nan"),
    )


def main():
    kw = dict(a.lstrip("-").split("=") for a in sys.argv[1:] if "=" in a)
    only = kw.get("only", "aadb,koniq").split(",")
    steps = int(kw.get("steps", 4000))
    desktop = kw.get("desktop", os.path.join(os.environ.get("PHOTOCOACH_DATA", os.path.join(ROOT, "data")), "features"))
    encs = kw.get("encoders", "mobileclip2_s0,mobileclip2_s2").split(",")
    runs = []
    if os.path.isdir(desktop):
        runs.append(("desktop ViT-L/14 + B/16", dict(feat=desktop, g_key="g")))
    for e in encs:
        f = os.path.join(MOB, e)
        if not all(os.path.exists(os.path.join(f, f"{n}.npz")) for n in only):
            print(f"(skipping {e}: features missing for {only})")
            continue
        runs += [
            (f"{e}  squash + crops + all signals", dict(feat=f, g_key="g_sq")),
            (f"{e}  letterbox + crops + all signals", dict(feat=f, g_key="g_lb")),
            (f"{e}  squash, no crops, all signals", dict(feat=f, g_key="g_sq", use_l=False)),
            (f"{e}  squash + crops + easy signals", dict(feat=f, g_key="g_sq", h_keys=EASY)),
            (f"{e}  squash + crops + size only", dict(feat=f, g_key="g_sq", h_keys=SIZE)),
        ]
    rows = []
    for name, cfg in runs:
        res = [summarise(train.main(steps=steps, only=only, quiet=True, seed=s,
                                    out=os.path.join(ROOT, "data", "features_mobile", "_pilot.pt"), **cfg)) for s in (0, 1)]
        rows.append((name, {k: float(np.mean([r[k] for r in res])) for k in res[0]}))
    cols = [c for c in rows[0][1] if not all(np.isnan(r[1][c]) for r in rows)]
    print("\n" + f"{'config (mean of 2 seeds)':46s}" + "".join(f"{c:>10s}" for c in cols))
    for name, r in rows:
        print(f"{name:46s}" + "".join(f"{r[c]:10.3f}" for c in cols))
    print("\nava/aadb/koniq/spaq: rank correlation with human ratings on held-out photos.  attrs: mean over the 11 AADB attributes."
          "\nspaq_attr: mean over exposure/contrast/sharpness/noise.  syn / syn_attr: clean photo beats its faulted copy / on the right attribute.")


if __name__ == "__main__":
    main()
