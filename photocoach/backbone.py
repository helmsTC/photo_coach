"""Frozen vision backbones + the image views fed to them.

Two views per photo:

  global view  - the WHOLE frame at its native aspect ratio (no centre crop, no
                 squashing; ViT position embeddings are interpolated), encoded by
                 CLIP ViT-L/14.  Carries composition, light, colour, subject.
  local views  - four 224px crops at native-ish resolution (long side 1024) placed
                 on the rule-of-thirds points, encoded by CLIP ViT-B/16.  Carry what
                 a downscaled global view cannot: micro-sharpness, noise, bokeh.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import cv2
from PIL import Image, ImageOps

GLOBAL_MODEL = "openai/clip-vit-large-patch14"
LOCAL_MODEL = "openai/clip-vit-base-patch16"
G_DIM, L_DIM, N_LOCAL = 768, 512, 4
PATCH = 14
TOKENS = 272
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


def load_image(src, max_side: int = 2048) -> Image.Image:
    """Open path / bytes / file-like, honour EXIF rotation, return RGB."""
    if isinstance(src, (bytes, bytearray)):
        import io
        src = io.BytesIO(src)
    im = Image.open(src)
    try:
        im.draft("RGB", (max_side, max_side))  # fast reduced-size JPEG decode
    except Exception:
        pass
    im = ImageOps.exif_transpose(im)
    if im.mode != "RGB":
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    return im


def grid_for(w: int, h: int, tokens: int = TOKENS) -> tuple[int, int]:
    a = min(2.0, max(0.5, w / h))
    return max(8, round(math.sqrt(tokens / a))), max(8, round(math.sqrt(tokens * a)))  # (rows, cols)


def global_view(rgb: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 -> uint8 (rows*14)x(cols*14)x3 covering the full frame."""
    h, w = rgb.shape[:2]
    gh, gw = grid_for(w, h)
    interp = cv2.INTER_AREA if (gw * PATCH < w) else cv2.INTER_CUBIC
    return cv2.resize(rgb, (gw * PATCH, gh * PATCH), interpolation=interp)


def local_views(rgb_std: np.ndarray, size: int = 224) -> np.ndarray:
    """Four native-resolution crops centred on the thirds intersections -> (4,size,size,3)."""
    h, w = rgb_std.shape[:2]
    if min(h, w) < size:
        s = size / min(h, w)
        rgb_std = cv2.resize(rgb_std, (max(size, round(w * s)), max(size, round(h * s))), interpolation=cv2.INTER_CUBIC)
        h, w = rgb_std.shape[:2]
    out = []
    for fy in (1 / 3, 2 / 3):
        for fx in (1 / 3, 2 / 3):
            y0 = int(min(max(fy * h - size / 2, 0), h - size))
            x0 = int(min(max(fx * w - size / 2, 0), w - size))
            out.append(rgb_std[y0:y0 + size, x0:x0 + size])
    return np.stack(out)


class Backbone:
    def __init__(self, device: str | None = None, load_local: bool = True):
        from transformers import CLIPVisionModelWithProjection
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.g = CLIPVisionModelWithProjection.from_pretrained(GLOBAL_MODEL).to(self.device).eval()
        self.l = CLIPVisionModelWithProjection.from_pretrained(LOCAL_MODEL).to(self.device).eval() if load_local else None
        for m in (self.g, self.l):
            if m is not None:
                m.requires_grad_(False)
        self.mean = torch.tensor(MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=self.device).view(1, 3, 1, 1)

    def _norm(self, u8: np.ndarray | torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(u8).to(self.device).permute(0, 3, 1, 2).float().div_(255.0)
        return (x - self.mean) / self.std

    @torch.inference_mode()
    def embed_global(self, u8_batch, bs: int = 32) -> torch.Tensor:
        """(N,H,W,3) uint8, all the same size -> (N,768) float32 on CPU."""
        out = []
        for i in range(0, len(u8_batch), bs):
            x = self._norm(u8_batch[i:i + bs])
            out.append(self.g(pixel_values=x, interpolate_pos_encoding=True).image_embeds.float().cpu())
        return torch.cat(out)

    @torch.inference_mode()
    def embed_local(self, u8_batch, bs: int = 96) -> torch.Tensor:
        """(N,224,224,3) uint8 -> (N,512)."""
        out = []
        for i in range(0, len(u8_batch), bs):
            x = self._norm(u8_batch[i:i + bs])
            out.append(self.l(pixel_values=x).image_embeds.float().cpu())
        return torch.cat(out)

    def embed_photo(self, rgb_std: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        g = self.embed_global(global_view(rgb_std)[None])[0]
        l = self.embed_local(local_views(rgb_std))
        return g, l
