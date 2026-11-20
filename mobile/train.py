"""Train the on-device judge: same twin tower and recipe as the desktop model, on features
from one phone-sized encoder.

    python -m mobile.train [--encoder=mobileclip2_s0] [--signals=easy|all|size] [--steps=9000]

Writes models/mobile_judge.pt (+ .report.json), which `python -m mobile.export_coreml` packages.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from photocoach import train
from photocoach.signals import VEC_KEYS

from .pilot import EASY, MOB, ROOT, SIZE

SIGNALS = {"all": list(VEC_KEYS), "easy": EASY, "size": SIZE}


def main():
    kw = dict(a.lstrip("-").split("=") for a in sys.argv[1:] if "=" in a)
    enc = kw.get("encoder", "mobileclip2_s0")
    sig = kw.get("signals", "easy")
    feat = os.path.join(MOB, enc)
    # the view size the features were cut at (the encoder's input size); the exporters and the apps use it
    with np.load(os.path.join(feat, "aadb.npz")) as d:
        res = int(d["res"]) if "res" in d.files else 256
    train.main(steps=int(kw.get("steps", 9000)), feat=feat, g_key="g_sq", h_keys=SIGNALS[sig],
               out=os.path.join(ROOT, "models", "mobile_judge.pt"), extra_meta=dict(encoder=enc, signals=sig, res=res))


if __name__ == "__main__":
    main()
