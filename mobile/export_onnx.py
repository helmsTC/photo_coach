"""Package the on-device models for the Android app (ONNX Runtime).

    python -m mobile.export_onnx                 # all four networks
    python -m mobile.export_onnx encoder tower   # a subset

Writes into android/app/src/main/assets/:
    encoder.onnx       image batch [n,3,S,S] RGB 0-255 -> embeddings (S = inputSize in judge.json)        (5 views per photo)
    tower.onnx         global embed + 4 crop embeds + signals -> 20 raw attribute scores
    segmenter.onnx     [1,3,448,448] RGB 0-255 -> COCO-panoptic class id per pixel (DETR-ResNet50)
    segmenter.json     class names and which ids are countable things
    ai_detector.onnx   [1,3,384,384] RGB 0-255 -> P(AI-generated)  (Community Forensics ViT-S/16-384)
    judge.json, parity.json, parity_*.jpg, sample_*.jpg   copied from the iOS resources (same model)

Weights are stored as 16-bit floats behind Cast nodes, which ONNX Runtime folds back to 32-bit when
the model loads: half the download, ordinary float32 inference on any CPU.

Unlike Core ML, ONNX Runtime runs on Linux, so each export is checked here against PyTorch, and
android/core's unit tests check the Kotlin pipeline against parity.json.
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "models", "mobile_judge.pt")
IOS = os.path.join(ROOT, "ios", "PhotoCoach", "Resources")
OUT = os.path.join(ROOT, "android", "app", "src", "main", "assets")
SEG = 448


def store_fp16(path: str, min_elems: int = 256) -> None:
    """Rewrite large float32 initializers as float16 + Cast(to float32)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    m = onnx.load(path)
    casts = []
    for init in m.graph.initializer:
        if init.data_type != TensorProto.FLOAT or int(np.prod(init.dims)) < min_elems:
            continue
        arr = numpy_helper.to_array(init)
        if not np.isfinite(arr).all() or np.abs(arr).max() > 6e4:
            continue
        name = init.name
        init.CopyFrom(numpy_helper.from_array(arr.astype(np.float16), name + "__f16"))
        casts.append(helper.make_node("Cast", [name + "__f16"], [name], to=TensorProto.FLOAT, name=name + "__cast"))
    nodes = casts + list(m.graph.node)
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    onnx.save(m, path)


def store_q8(path: str, min_elems: int = 2048) -> float:
    """Rewrite large float32 weight tensors as int8 x a float scale per slice (Cast + Mul, folded when
    the session loads): a quarter of float32.  The slicing axis is whichever of first/last reproduces
    the tensor best, so weights that are already on an 8-bit grid (an encoder loaded through
    mobile.extract.quantize8) are stored exactly whichever way the exporter laid them out.
    Returns the largest rounding error relative to the tensor's own largest value."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    m = onnx.load(path)
    new, extra, worst = [], [], 0.0
    for init in m.graph.initializer:
        if init.data_type != TensorProto.FLOAT or len(init.dims) < 2 or int(np.prod(init.dims)) < min_elems:
            continue
        w = numpy_helper.to_array(init)
        if not np.isfinite(w).all():
            continue
        best = None
        for axis in {0, w.ndim - 1}:
            other = tuple(i for i in range(w.ndim) if i != axis)
            sc = (np.maximum(np.abs(w).max(axis=other, keepdims=True), 1e-12) / 127).astype(np.float32)
            q = np.clip(np.rint(w / sc), -127, 127).astype(np.int8)
            err = float(np.abs(q.astype(np.float32) * sc - w).max() / max(float(np.abs(w).max()), 1e-12))
            if best is None or err < best[0]:
                best = (err, q, sc)
        err, q, sc = best
        worst = max(worst, err)
        name = init.name
        init.CopyFrom(numpy_helper.from_array(q, name + "__q8"))
        extra.append(numpy_helper.from_array(sc, name + "__scale"))
        new.append(helper.make_node("Cast", [name + "__q8"], [name + "__q8f"], to=TensorProto.FLOAT, name=name + "__q8cast"))
        new.append(helper.make_node("Mul", [name + "__q8f", name + "__scale"], [name], name=name + "__q8mul"))
    m.graph.initializer.extend(extra)
    nodes = new + list(m.graph.node)
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    onnx.save(m, path)
    return worst


def _palette(x: np.ndarray, k: int = 256, iters: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """1-D k-means: -> (k centres, index of each value)."""
    x = x.ravel().astype(np.float64)
    c = np.unique(np.concatenate([np.quantile(x, (np.arange(k) + 0.5) / k)[::2], np.linspace(x.min(), x.max(), k)[::2]]))[:k]
    for _ in range(iters):
        idx = np.searchsorted((c[1:] + c[:-1]) / 2, x)
        n = np.bincount(idx, minlength=len(c))
        c = np.sort(np.where(n > 0, np.bincount(idx, weights=x, minlength=len(c)) / np.maximum(n, 1), c))
    idx = np.searchsorted((c[1:] + c[:-1]) / 2, x)
    lut = np.zeros(k, np.float32)
    lut[:len(c)] = c
    return lut, idx.astype(np.uint8)


def store_pal8(path: str, min_elems: int = 2048) -> None:
    """Rewrite large float32 weight tensors as 8-bit indices into a 256-value table, times a scale per
    slice along axis 0 (Cast + Gather + Mul).  Lossy, but much gentler than linear 8-bit on
    convolutions with folded batch-norm: DETR's labels survive this and do not survive store_q8."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    m = onnx.load(path)
    new, extra = [], []
    for init in m.graph.initializer:
        if init.data_type != TensorProto.FLOAT or len(init.dims) < 2 or int(np.prod(init.dims)) < min_elems:
            continue
        w = numpy_helper.to_array(init)
        if not np.isfinite(w).all():
            continue
        sc = np.maximum(np.abs(w).reshape(w.shape[0], -1).max(1), 1e-12).reshape(-1, *[1] * (w.ndim - 1)).astype(np.float32)
        lut, idx = _palette(w / sc)
        name = init.name
        init.CopyFrom(numpy_helper.from_array(idx.reshape(w.shape), name + "__p8"))
        extra += [numpy_helper.from_array(lut, name + "__lut"), numpy_helper.from_array(sc, name + "__scale")]
        new.append(helper.make_node("Cast", [name + "__p8"], [name + "__p8i"], to=TensorProto.INT32, name=name + "__p8cast"))
        new.append(helper.make_node("Gather", [name + "__lut", name + "__p8i"], [name + "__p8f"], axis=0, name=name + "__p8gather"))
        new.append(helper.make_node("Mul", [name + "__p8f", name + "__scale"], [name], name=name + "__p8mul"))
    m.graph.initializer.extend(extra)
    nodes = new + list(m.graph.node)
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    onnx.save(m, path)


def _mb(p):
    return os.path.getsize(p) / 1e6


def _session(path):
    import onnxruntime as ort
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def _export(module, args, path, inputs, outputs, dynamic=None, opset=17):
    torch.onnx.export(module, args, path, input_names=inputs, output_names=outputs, dynamic_axes=dynamic or {},
                      opset_version=opset, do_constant_folding=True, dynamo=False)


# ---------------------------------------------------------------- encoder + tower

def export_encoder():
    from timm.utils import reparameterize_model
    from photocoach.model import load_judge
    from .export_coreml import EncoderWrap
    from .extract import RES, load_visual

    _, meta = load_judge(CKPT)
    name = meta["encoder"]
    visual, size, mean, std = load_visual(name)
    res = int(meta.get("res", RES))                       # the view size the tower's features were cut at
    assert size == res, f"{name} takes {size}px input but the features were extracted at {res}px"
    if name.startswith("mobileclip"):
        visual = reparameterize_model(visual.eval())
    enc = EncoderWrap(visual.eval(), mean, std).eval()
    ex = torch.rand(3, 3, res, res) * 255

    p = os.path.join(OUT, "encoder.onnx")
    _export(enc, (ex,), p, ["image"], ["embedding"], {"image": {0: "n"}, "embedding": {0: "n"}})
    if name.endswith("_q8"):                              # weights are already on an 8-bit grid: store them as such
        print(f"   8-bit storage: worst rounding error {store_q8(p):.1e} of a tensor's range (0 = exact)")
    store_fp16(p)
    with torch.no_grad():
        ref = enc(ex).numpy()
    got = _session(p).run(None, {"image": ex.numpy()})[0]
    cos = (ref * got).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(got, axis=1))
    print(f"encoder.onnx      {_mb(p):6.1f} MB   max |diff| {np.abs(ref - got).max():.4f} (values up to {np.abs(ref).max():.2f}), cosine >= {cos.min():.6f}")


def export_tower():
    from photocoach.model import load_judge
    from photocoach.signals import VEC_KEYS
    from .export_coreml import TowerWrap

    judge, meta = load_judge(CKPT)
    arch = meta["arch"]
    tw = TowerWrap(judge.tower).eval()
    eg, el, eh = torch.randn(1, arch["g_dim"]), torch.randn(1, 4, arch["l_dim"]), torch.zeros(1, len(VEC_KEYS))
    p = os.path.join(OUT, "tower.onnx")
    _export(tw, (eg, el, eh), p, ["g", "l", "h"], ["scores"])
    with torch.no_grad():
        ref = judge.tower(eg, el, eh, torch.ones(1)).numpy()
    got = _session(p).run(None, {"g": eg.numpy(), "l": el.numpy(), "h": eh.numpy()})[0]
    print(f"tower.onnx        {_mb(p):6.1f} MB   max |diff| {np.abs(ref - got).max():.6f}")


# ---------------------------------------------------------------- segmenter

class SegWrap(nn.Module):
    """DETR panoptic head -> one COCO-panoptic class id per pixel (the usual semantic post-processing:
    class probabilities x mask probabilities, summed over queries, arg-max over classes)."""

    def __init__(self, detr):
        super().__init__()
        self.detr = detr
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255.0)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255.0)

    def forward(self, image):
        x = (image - self.mean) / self.std
        out = self.detr(pixel_values=x, pixel_mask=torch.ones(x.shape[0], x.shape[2], x.shape[3], dtype=torch.long))
        cls = out.logits.softmax(-1)[..., :-1]                     # [1, Q, C]   (last = no object)
        masks = out.pred_masks.sigmoid()                           # [1, Q, h, w]
        seg = torch.einsum("bqc,bqhw->bchw", cls, masks)
        seg = nn.functional.interpolate(seg, size=(SEG, SEG), mode="bilinear", align_corners=False)
        return seg.argmax(1).to(torch.int32).reshape(-1)


# COCO panoptic "stuff" categories (panoptic_coco_categories.json).  The checkpoint's own config names
# only some of them and leaves the merged ones as LABEL_n.
STUFF = {
    92: "banner", 93: "blanket", 95: "bridge", 100: "cardboard", 107: "counter", 109: "curtain", 112: "door-stuff",
    118: "floor-wood", 119: "flower", 122: "fruit", 125: "gravel", 128: "house", 130: "light", 133: "mirror-stuff",
    138: "net", 141: "pillow", 144: "platform", 145: "playingfield", 147: "railroad", 148: "river", 149: "road",
    151: "roof", 154: "sand", 155: "sea", 156: "shelf", 159: "snow", 161: "stairs", 166: "tent", 168: "towel",
    171: "wall-brick", 175: "wall-stone", 176: "wall-tile", 177: "wall-wood", 178: "water-other", 180: "window-blind",
    181: "window-other", 184: "tree-merged", 185: "fence-merged", 186: "ceiling-merged", 187: "sky-other-merged",
    188: "cabinet-merged", 189: "table-merged", 190: "floor-other-merged", 191: "pavement-merged", 192: "mountain-merged",
    193: "grass-merged", 194: "dirt-merged", 195: "paper-merged", 196: "food-other-merged", 197: "building-other-merged",
    198: "rock-merged", 199: "wall-other-merged", 200: "rug-merged",
}


def _stuff_label(raw: str) -> str:
    """'sky-other-merged' -> 'sky (other)', 'tree-merged' -> 'tree': the style the app's naming expects."""
    n = raw[:-7] if raw.endswith("-merged") else raw
    if "-" in n:
        a, b = n.split("-", 1)
        return f"{a} ({b})"
    return n


def seg_labels(id2label: dict) -> tuple[list, list]:
    n = max(max(id2label), max(STUFF)) + 1
    things = [i for i in range(1, 91) if i in id2label and not id2label[i].startswith(("N/A", "LABEL_"))]
    labels = ["--"] * n
    for i in things:
        labels[i] = id2label[i]
    for i, name in STUFF.items():
        labels[i] = _stuff_label(name)
    return labels, things


def export_segmenter():
    from transformers import DetrForSegmentation

    detr = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic").eval()
    id2label = {int(k): v for k, v in detr.config.id2label.items()}
    labels, things = seg_labels(id2label)
    w = SegWrap(detr).eval()
    ex = torch.rand(1, 3, SEG, SEG) * 255
    p = os.path.join(OUT, "segmenter.onnx")
    _export(w, (ex,), p, ["image"], ["classes"])
    store_pal8(p)
    store_fp16(p)
    json.dump(dict(size=SEG, labels=labels, things=things, model="facebook/detr-resnet-50-panoptic (Apache-2.0)"),
              open(os.path.join(OUT, "segmenter.json"), "w"))

    # check on real photos: the compressed ONNX vs PyTorch
    import cv2
    from photocoach.backbone import load_image
    sess = _session(p)
    agree = []
    for f in ("louise_03.jpg", "zclutter_table.jpg", "zclutter_stage.jpg"):
        rgb = np.asarray(load_image(os.path.join(ROOT, "data", "demo", f)))
        sq = cv2.resize(rgb, (SEG, SEG), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(sq).permute(2, 0, 1)[None].float()
        with torch.no_grad():
            ref = w(x).numpy()
        got = sess.run(None, {"image": x.numpy()})[0]
        agree.append((ref == got).mean())
        ids, cnt = np.unique(got, return_counts=True)
        top = ", ".join(f"{labels[i]} {c / got.size:.0%}" for i, c in sorted(zip(ids, cnt), key=lambda t: -t[1])[:5])
        print(f"   {f}: {top}")
    print(f"segmenter.onnx    {_mb(p):6.1f} MB   pixel agreement with PyTorch >= {min(agree):.4f}")


# ---------------------------------------------------------------- AI detector

class DetWrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255)

    def forward(self, x):          # x: RGB 0-255, [1,3,384,384]
        return torch.sigmoid(self.m((x - self.mean) / self.std)).reshape(-1)


def load_detector():
    import timm
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    vit = timm.create_model("vit_small_patch16_384.augreg_in21k_ft_in1k", pretrained=False)
    vit.head = nn.Linear(384, 1)
    sd = load_file(hf_hub_download("OwensLab/commfor-model-384", "model.safetensors"))
    vit.load_state_dict({k.removeprefix("vit."): v for k, v in sd.items()})
    return DetWrap(vit.eval()).eval()


def detector_input(rgb: np.ndarray) -> torch.Tensor:
    """Reference preprocessing (PIL): shorter side to 440 bicubic, centre crop 384."""
    from PIL import Image
    im = Image.fromarray(rgb)
    k = 440 / min(im.size)
    w, h = max(384, round(im.size[0] * k)), max(384, round(im.size[1] * k))
    im = im.resize((w, h), Image.BICUBIC)
    x0, y0 = round((w - 384) / 2), round((h - 384) / 2)
    a = np.asarray(im.crop((x0, y0, x0 + 384, y0 + 384)), dtype=np.float32)
    return torch.from_numpy(a).permute(2, 0, 1)[None]


def export_detector():
    w = load_detector()
    ex = torch.rand(1, 3, 384, 384) * 255
    p = os.path.join(OUT, "ai_detector.onnx")
    _export(w, (ex,), p, ["image"], ["ai"])
    store_fp16(p)             # 16-bit only: 8-bit weights changed verdicts on a test set (see export_ai_detector.py)
    from photocoach.backbone import load_image
    sess = _session(p)
    worst = 0.0
    for f in ("louise_03.jpg", "louise_11.jpg", "zclutter_table.jpg"):
        x = detector_input(np.asarray(load_image(os.path.join(ROOT, "data", "demo", f))))
        with torch.no_grad():
            ref = float(w(x)[0])
        got = float(sess.run(None, {"image": x.numpy()})[0][0])
        worst = max(worst, abs(ref - got))
        print(f"   {f}: P(AI) torch {ref:.4f}  onnx {got:.4f}")
    print(f"ai_detector.onnx  {_mb(p):6.1f} MB   max |diff| {worst:.4f}")


def copy_shared():
    n = 0
    for f in sorted(os.listdir(IOS)):
        if f in ("judge.json", "parity.json") or (f.endswith(".jpg") and f.split("_")[0] in ("parity", "sample")):
            shutil.copyfile(os.path.join(IOS, f), os.path.join(OUT, f))
            n += 1
    print(f"copied {n} shared files (judge.json, parity fixtures, samples) from the iOS resources")


STEPS = dict(encoder=export_encoder, tower=export_tower, segmenter=export_segmenter, detector=export_detector, shared=copy_shared)


def main(which):
    os.makedirs(OUT, exist_ok=True)
    for k in which or list(STEPS):
        STEPS[k]()


if __name__ == "__main__":
    main(sys.argv[1:])
