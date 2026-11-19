"""Sanity checks for the trained system.   Run:  python -m pytest tests -q   (or python tests/test_photocoach.py)

They need the trained checkpoint (models/twin_judge.pt) and a GPU or some patience.
"""
import glob
import io
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from photocoach import signals, synth  # noqa: E402
from photocoach.backbone import load_image  # noqa: E402
from photocoach.engine import Engine  # noqa: E402

_E = None


def eng():
    global _E
    if _E is None:
        _E = Engine()
    return _E


def _jpg(rgb):
    b = io.BytesIO()
    Image.fromarray(rgb).save(b, "JPEG", quality=93)
    return b.getvalue()


def _sources(n=6):
    fs = sorted(glob.glob(os.path.join(ROOT, "data", "demo", "*.jpg")))[:n]
    assert fs, "put a few photos in data/demo"
    return [signals.to_std(np.asarray(load_image(f))) for f in fs]


def test_twin_is_antisymmetric():
    a, b = _sources(2)
    r1, _ = eng().compare(_jpg(a), _jpg(b), "A", "B", deep=False)
    r2, _ = eng().compare(_jpg(b), _jpg(a), "B", "A", deep=False)
    assert abs(r1["p_a"] + r2["p_a"] - 1.0) < 1e-3
    assert r1["photos"][r1["winner"]]["name"] == r2["photos"][r2["winner"]]["name"]


def test_identical_photos_are_a_tie():
    a = _sources(1)[0]
    r, _ = eng().compare(_jpg(a), _jpg(a), "A", "B", deep=False)
    assert abs(r["p_a"] - 0.5) < 0.02 and r["close_call"]
    assert not r["coaching"], [c["title"] for c in r["coaching"]]


def test_clean_original_beats_its_faulted_copy_and_names_the_fault():
    rng = np.random.default_rng(0)
    cases = {"overexpose": ("highlights", "brightness"), "underexpose": ("brightness", "shadows"),
             "defocus": ("sharpness",), "noise": ("noise",), "flat_contrast": ("contrast",), "desaturate": ("muted",)}
    wins = named = total = 0
    for src in _sources(4):
        for eff, ids in cases.items():
            bad = synth.degrade(src, eff, 0.7, rng)
            r, _ = eng().compare(_jpg(bad), _jpg(src), "faulty", "clean", deep=False)
            total += 1
            wins += r["winner"] == "b"
            named += any(c["id"] in ids for c in r["coaching"][:3])
    assert wins / total >= 0.9, f"clean photo won only {wins}/{total}"
    assert named / total >= 0.75, f"fault named in top-3 advice only {named}/{total}"


def test_same_scene_crop_is_located():
    src = _sources(1)[0]
    h, w = src.shape[:2]
    tight = src[int(.2 * h):int(.85 * h), int(.25 * w):int(.9 * w)]
    r, imgs = eng().compare(_jpg(src), _jpg(tight), "wide", "tight", deep=False)
    assert r["relation"]["kind"] == "aligned"
    assert any(k.endswith("_frame") for k in imgs)


def test_objects_are_found_and_a_known_distraction_is_flagged():
    """The stage photo has a second figure cut off at the bottom right; removing it raises the score."""
    f = os.path.join(ROOT, "data", "demo", "zclutter_stage.jpg")
    g = os.path.join(ROOT, "data", "demo", "louise_11.jpg")
    r, imgs = eng().compare(open(f, "rb").read(), open(g, "rb").read(), "stage", "lake", deep=True)
    assert "a_objects" in imgs and "b_objects" in imgs
    objs = r["photos"]["a"]["objects"]
    assert r["photos"]["a"]["scene"]["subject"]["label"] == "person"
    assert any(o["role"] == "distraction" and o["label"] == "person" for o in objs), objs
    assert {"water", "sky"} <= {o["label"] for o in r["photos"]["b"]["objects"]}
    if r["loser"] == "a":
        assert any(c["id"] == "distraction" for c in r["coaching"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
