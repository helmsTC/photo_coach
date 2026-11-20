"""Cache features from phone-sized backbones for every training image.

    PHOTOCOACH_DATA=/path/to/data python -m mobile.extract aadb koniq pros spaq ava [--limit=N]

Unlike the desktop pipeline (variable-shape ViT-L/14 + a second encoder), the on-device
design uses ONE small image encoder with a fixed square input, run five times per photo:
once on the whole frame and once on each of four native-resolution crops.  Two ways of
fitting the whole frame into the square are cached so they can be compared:

    g_sq  squashed (aspect ratio lost, every pixel used)
    g_lb  letterboxed (aspect ratio kept, grey bars)

Several candidate encoders are run in the same pass over the images.  Each one gets views cut at
its OWN input size (which is what the apps do: they read `inputSize` from judge.json), so a 224 px
encoder sees 224 px crops rather than 256 px crops resized.
Output: data/features_mobile/<encoder>/<dataset>.npz   (with `res`, the view size used)
"""
from __future__ import annotations

import os
import sys
import time
import zlib
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import cv2
import numpy as np
import torch

from photocoach import signals, synth
from photocoach.backbone import load_image, local_views
from photocoach.extract import SOURCES, _init_worker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "features_mobile")
RES = 256
CANDIDATES = {                       # name -> (open_clip model, pretrained tag), or ("timm:<model>", None)
    # Apple ML Research Model License (non-commercial): kept only as the accuracy baseline
    "mobileclip2_s0": ("MobileCLIP2-S0", "dfndr2b"),
    "mobileclip2_s2": ("MobileCLIP2-S2", "dfndr2b"),
    "mobileclip2_b":  ("MobileCLIP2-B", "dfndr2b"),
    # commercially usable
    "clip_b32":       ("ViT-B-32-quickgelu", "openai"),             # MIT
    "siglip2_b16":    ("ViT-B-16-SigLIP2-256", "webli"),            # Apache 2.0
    "siglip2_b32":    ("ViT-B-32-SigLIP2-256", "webli"),            # Apache 2.0, a quarter of the tokens of B/16
    "dinov2_s14":     ("timm:vit_small_patch14_dinov2.lvd142m", None),   # Apache 2.0
}
# what each encoder is called in judge.json / model descriptions
DISPLAY = {"clip_b32": "OpenAI CLIP ViT-B/32", "siglip2_b16": "SigLIP 2 ViT-B/16", "siglip2_b32": "SigLIP 2 ViT-B/32",
           "dinov2_s14": "DINOv2 ViT-S/14"}


def load_visual(name: str):
    """-> (image tower returning one embedding per image, input size, mean, std)."""
    arch, tag = CANDIDATES[name]
    if arch.startswith("timm:"):
        import timm
        m = timm.create_model(arch[5:], pretrained=True, num_classes=0, img_size=224)   # CLS embedding
        return m, 224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    import open_clip
    m = open_clip.create_model(arch, pretrained=tag)
    cfg = open_clip.get_model_preprocess_cfg(m)
    size = cfg["size"] if isinstance(cfg["size"], int) else cfg["size"][0]
    return m.visual, size, tuple(cfg["mean"]), tuple(cfg["std"])


def squash(rgb: np.ndarray, res: int = RES) -> np.ndarray:
    return cv2.resize(rgb, (res, res), interpolation=cv2.INTER_AREA)


def letterbox(rgb: np.ndarray, res: int = RES) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = res / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    out = np.full((res, res, 3), 124, np.uint8)
    y0, x0 = (res - nh) // 2, (res - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    return out


def views(rgb_std: np.ndarray, res: int = RES) -> np.ndarray:
    """(6,res,res,3): squashed frame, letterboxed frame, four native-resolution crops at the thirds points."""
    return np.concatenate([squash(rgb_std, res)[None], letterbox(rgb_std, res)[None], local_views(rgb_std, res)])


class Encoder:
    def __init__(self, name: str, device: str):
        visual, size, mean, std = load_visual(name)
        self.size = size                # the app cuts its views at this size, and so does extraction
        self.visual = visual.to(device).eval().requires_grad_(False)
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        self.device = device
        self.name = name
        print(f"  {name}: input {size}px, {sum(p.numel() for p in self.visual.parameters()) / 1e6:.1f}M parameters", flush=True)

    @torch.inference_mode()
    def embed(self, u8: np.ndarray, bs: int = 64) -> torch.Tensor:
        """(N,size,size,3) uint8 -> (N,D) float32 on CPU (not normalised)."""
        out = []
        for i in range(0, len(u8), bs):
            x = torch.as_tensor(u8[i:i + bs]).to(self.device).permute(0, 3, 1, 2).float().div_(255.0)
            assert x.shape[-1] == self.size, f"{self.name} expects {self.size}px views, got {x.shape[-1]}"
            out.append(self.visual((x - self.mean) / self.std).float().cpu())
        return torch.cat(out)


def _prep(args):
    key, data, meta, sizes = args
    try:
        rgb = signals.to_std(np.asarray(load_image(data, max_side=1536)))
        if min(rgb.shape[:2]) < 48:
            return None
        variants = [("", 0.0, rgb)]
        if meta["n_synth"]:
            rng = np.random.default_rng(zlib.crc32(key.encode()))
            variants += synth.random_variants(rgb, meta["n_synth"], rng)
        return [dict(key=key, effect=eff, severity=sev, meta=meta, v={r: views(im, r) for r in sizes},
                     aspect=im.shape[1] / im.shape[0],
                     h=signals.vector(signals.measure(im, full=False))) for eff, sev, im in variants]
    except Exception as e:
        print(f"  skip {key}: {e}", flush=True)
        return None


def run(name: str, encoders: list, limit: int | None = None):
    sizes = sorted({e.size for e in encoders})
    recs = []
    buf = {r: [] for r in sizes}
    F = {e.name: [] for e in encoders}

    def flush():
        n = len(buf[sizes[0]])              # records waiting (each holds 6 views)
        if not n:
            return
        for r in sizes:
            x = np.concatenate(buf[r])
            for e in encoders:
                if e.size == r:
                    F[e.name].append(e.embed(x).view(n, 6, -1).numpy().astype(np.float16))
            buf[r].clear()

    def save():
        flush()
        label_keys = sorted({k for r in recs for k in r["meta"] if k not in ("split", "local", "n_synth")})
        base = dict(key=np.array([r["key"] for r in recs]), effect=np.array([r["effect"] for r in recs]),
                    severity=np.array([r["severity"] for r in recs], np.float32),
                    split=np.array([r["meta"]["split"] for r in recs]),
                    aspect=np.array([r["aspect"] for r in recs], np.float32),
                    h=np.stack([r["h"] for r in recs]))
        for k in label_keys:
            base["y_" + k] = np.array([r["meta"].get(k, np.nan) for r in recs], np.float32)
        for e in encoders:
            f = np.concatenate(F[e.name])
            d = os.path.join(OUT, e.name)
            os.makedirs(d, exist_ok=True)
            tmp = os.path.join(d, f"{name}.tmp.npz")
            np.savez(tmp, g_sq=f[:, 0], g_lb=f[:, 1], l=f[:, 2:], res=np.int32(e.size), **base)
            os.replace(tmp, os.path.join(d, f"{name}.npz"))
        print(f"[{name}] saved {len(recs)} records for {len(encoders)} encoders", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(18, mp_context=mp.get_context("spawn"), initializer=_init_worker) as ex:
        pending = deque()
        n_in = 0

        def drain(block_all=False):
            while pending and (block_all or len(pending) >= 160 or pending[0].done()):
                for r in pending.popleft().result() or []:
                    v = r.pop("v")
                    for sz in sizes:
                        buf[sz].append(v[sz])
                    recs.append(r)
                    if len(buf[sizes[0]]) >= 64:
                        flush()
                    if len(recs) % 2000 == 0:
                        el = time.time() - t0
                        print(f"[{name}] {len(recs)} records  {len(recs) / el:.1f}/s  {el / 60:.1f} min", flush=True)

        for item in SOURCES[name]():
            pending.append(ex.submit(_prep, (*item, sizes)))
            n_in += 1
            drain()
            if limit and n_in >= limit:
                break
        drain(block_all=True)
    save()
    print(f"[{name}] done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    limit = next((int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--limit=")), None)
    only = next((a.split("=")[1].split(",") for a in sys.argv[1:] if a.startswith("--encoders=")), list(CANDIDATES))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    encs = [Encoder(n, dev) for n in only]
    for nm in names:
        run(nm, encs, limit)
