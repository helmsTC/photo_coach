# Replacing MobileCLIP2 in the iOS judge

## Why

The on-device judge is our twin network: both photos go through the same **Encoder** (image → feature vector,
run 5× per photo: whole frame + 4 detail crops), then the same **Tower** (features + measured signals → 20
attribute scores), and the verdict compares the two sets of scores.

The Encoder is currently **MobileCLIP2-S0**, whose weights are under Apple's *ML Research Model License*:
non-commercial research only. That blocks selling ApertureIQ. The Tower, signals, coaching, AI detector
(Community Forensics, MIT) and object detection (DETR, Apache 2.0) are all fine.

**Fix:** swap the Encoder for a commercially licensed one and retrain the Tower on its features. The
twin design, the training recipe and the app code stay the same.

## The three candidates

| | OpenAI CLIP ViT-B/32 | SigLIP 2 ViT-B/16 | DINOv2 ViT-S/14 |
|---|---|---|---|
| License | **MIT** | **Apache 2.0** | **Apache 2.0** |
| Source to verify | huggingface.co/openai/clip-vit-base-patch32 | huggingface.co/google/siglip2-base-patch16-256 | huggingface.co/facebook/dinov2-small |
| Loader | open_clip `("ViT-B-32-quickgelu", "openai")` | open_clip `("ViT-B-16-SigLIP2-256", "webli")` | timm `vit_small_patch14_dinov2.lvd142m` |
| Input size | 224 | 256 (same as today) | 224 (must be a multiple of 14) |
| Embedding | 512 | 768 | 384 |
| Image params | ~88M | ~93M | ~22M |
| Encoder size, FP16 / 8-bit | ~176 MB / ~88 MB | ~186 MB / ~93 MB | ~44 MB / ~22 MB |
| Speed (5 views) | fast (49 tokens) | slowest (256 tokens) | fast (256 tokens, small model) |
| Notes | Same family as the desktop model (ViT-L/14 + B/16). Proven for aesthetics. Safest choice. | Newest; usually the strongest CLIP-style features. | Closest to today's size (MobileCLIP2-S0 is ~23 MB). Strong on image quality, weaker on content/meaning. |

Check the license on each model card yourself before shipping; the table is from memory.
Check the open_clip names exist in your installed version first:

```python
import open_clip; print([p for p in open_clip.list_pretrained() if "B-32" in p[0] or "SigLIP2" in p[0]])
```

**How to choose:** run the pilot (step 3) and pick the smallest encoder whose numbers match or beat
`mobileclip2_s0`, within about 0.01 SRCC on AVA/AADB/KonIQ/SPAQ and 1% on the synthetic pair accuracy. If
none match, take the most accurate and quantize it to 8-bit.

## Steps on the training machine

### 0. Setup

```bash
git fetch && git switch ios-clip-fix
git lfs install && git lfs pull          # model weights are in Git LFS
pip install -U open_clip_torch timm coremltools
export PHOTOCOACH_DATA=/path/to/data     # the five datasets, same as the original training run
```

### 1. Code changes

**`mobile/extract.py`**

- Add the candidates. Use a `timm:` prefix for the non-open_clip one:
  ```python
  CANDIDATES = {
      "mobileclip2_s0": ("MobileCLIP2-S0", "dfndr2b"),          # baseline for comparison only
      "clip_b32":       ("ViT-B-32-quickgelu", "openai"),
      "siglip2_b16":    ("ViT-B-16-SigLIP2-256", "webli"),
      "dinov2_s14":     ("timm:vit_small_patch14_dinov2.lvd142m", None),
  }
  ```
- In `Encoder.__init__`, branch on the `timm:` prefix:
  ```python
  if arch.startswith("timm:"):
      import timm
      m = timm.create_model(arch[5:], pretrained=True, num_classes=0, img_size=224)  # CLS embedding, 384-d
      self.visual, size = m, 224
      mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
  else:
      ...existing open_clip path...
  ```
- **Cut the views at the encoder's own input size** rather than cutting at 256 and resizing. That is
  what the app will do: it reads `inputSize` from `judge.json` and passes it to `ImagePrep.views`. Add a
  `--res=` argument (default 256) used by `views(...)`, and extract each size group in its own run, so
  every encoder's features come from views at its native size:
  ```bash
  python -m mobile.extract aadb koniq pros spaq ava --encoders=clip_b32,dinov2_s14 --res=224
  python -m mobile.extract aadb koniq pros spaq ava --encoders=siglip2_b16,mobileclip2_s0 --res=256
  ```
  (Skip `mobileclip2_s0` if its features are already cached in `data/features_mobile/mobileclip2_s0/`.)
  Store the `res` used in each `.npz`, so training and export can read it back.

**`mobile/export_coreml.py`**

- Load the encoder through the same `Encoder` helper (or the same `timm:` branch), not
  `open_clip.create_model` directly.
- Replace `assert size == RES` with the encoder's own size, read from the features/checkpoint meta. Use
  it for the Core ML input shape, the parity views and `inputSize` in `judge.json`.
- Only call `reparameterize_model` for MobileCLIP; it is harmless on the others, but unnecessary.
- Set the `ml.short_description` and `info["encoder"]` to the new model's name.
- Optional, for the big CLIP-style encoders: 8-bit weights, about half the size:
  ```python
  import coremltools.optimize.coreml as cto
  ml = cto.linear_quantize_weights(ml, cto.OptimizationConfig(
      global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric")))
  ```
  Re-check parity (step 5) after quantizing; keep FP16 if the scores move noticeably.

**No Swift changes are needed for the model itself.** `PhotoJudge.embedOnce` reads embeddings of any
length, the Tower takes whatever dimension it was trained on, and the view size comes from `judge.json`.

### 2. Extract features

Run the two `mobile.extract` commands above. Output: `data/features_mobile/<encoder>/<dataset>.npz`.

### 3. Pilot: compare the candidates

```bash
python -m mobile.pilot --encoders=mobileclip2_s0,clip_b32,siglip2_b16,dinov2_s14 --steps=4000
```

This trains the same twin tower on each encoder's features and prints the held-out numbers side by side:
AVA/AADB aesthetic SRCC, AADB attributes, KonIQ/SPAQ technical, SPAQ attributes, and synthetic pair accuracy.
Save the output and pick the encoder (see *How to choose*).

### 4. Train the final Tower

```bash
python -m mobile.train --encoder=<chosen> --signals=easy --steps=9000
```

This writes `models/mobile_judge.pt` and its `.report.json`.

### 5. Export for iOS

```bash
python -m mobile.export_coreml
```

This rewrites, in `ios/PhotoCoach/Resources/`:

- `Encoder.mlpackage`
- `Tower.mlpackage`
- `judge.json` (encoder name, `inputSize`, calibration and advice)
- `parity.json` and `parity_louise_*.jpg`

Commit and push them; the `.bin` weights go through LFS automatically.

```bash
git add -A ios/PhotoCoach/Resources models mobile && git commit -m "iOS: replace MobileCLIP2 encoder with <chosen>" && git push
```

## Steps back on the Mac

### 6. Rebuild and check parity

```bash
git pull && git lfs pull
cd ios && xcodegen generate
```

- Run a **Debug** build in the simulator. Under **⋯ › Run self-test**, every parity line must pass. The
  self-test compares Swift preprocessing + Core ML against the PyTorch reference numbers. You can also
  launch with `-diag`.
- Compare a few sample pairs before and after: winners should mostly agree with the old model.
- On an iPhone, time a Pro comparison and check the app's download size in Xcode's Organizer.

### 7. Update credits and docs

- `ios/PhotoCoach/Sources/UI/AboutView.swift`: remove the MobileCLIP2 credit and add the new encoder.
  - MIT (CLIP): include the full MIT text with `Copyright (c) 2021 OpenAI`, as is done for Community Forensics.
  - Apache 2.0 (SigLIP 2 / DINOv2): name the model, its authors and the Apache License 2.0.
- `ios/README.md`: replace the licensing warning with the new encoder and its license.
- Keep the Acknowledgements intro that explains the twin network is ApertureIQ's own.

### 8. Ship

Merge `ios-clip-fix` into `ios`. Then:

1. Archive a Release build.
2. Upload it to App Store Connect.
3. Start TestFlight, internal first, then external.

## Summary of what changes

| Changes | Stays the same |
|---|---|
| `Encoder.mlpackage`, `Tower.mlpackage`, `judge.json`, `parity.json` | Twin-network design and training recipe |
| `mobile/extract.py`, `mobile/export_coreml.py` (loader + input size) | All Swift judging, signals, coaching, UI |
| About screen credits, README licensing note | AI detector, object detection, history, StoreKit |
