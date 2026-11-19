# Photo Coach

Compare two photos with a twin network. It picks the stronger one, shows the
evidence, and tells you what to change to close the gap.

## Run it

```bash
pip install -r requirements.txt
python -m photocoach.app          # open http://127.0.0.1:5000
```

Drop in two photos (for example your shot and a professional's photo of the same
view) and press **Compare**. A comparison takes 10-30 s on a GTX 1080 Ti.

## What you get

| Section | What it shows |
|---|---|
| Verdict | Winner, how confident the model is, overall / aesthetic / technical scores as percentiles |
| Coaching | Prioritised list of what separates the photos, each with evidence, what to do at the shoot, what to do in the edit, and whether it is fixable in editing at all |
| Evidence overlays | Clipping (blown / crushed pixels), focus map, composition (thirds, visual weight, horizon), "what carries the score" (occlusion map), and - when both photos show the same scene - the winner's framing drawn inside yours |
| Objects in the frame | A panoptic segmentation model (Mask2Former Swin-S, COCO) outlines every object and background region. The main subject is picked from the masks, and each object is removed in turn and the photo re-scored by the twin network, so the app can say which objects carry the score and which are distractions ("with the person at the lower right painted out, the photo scores 11 points higher"). Subject size, cut-off at the frame edge, separation from the background and sky share are measured from the masks and compared between the two photos |
| Attribute breakdown | 16 attributes, both photos on one 0-100 scale |
| Edit preview | Your photo with the winner's tone/colour (and crop, if the scenes match) applied, re-scored by the network: how much of the gap is editing vs. needs a reshoot, plus the slider moves |
| Measurements | Histogram, palette, clipping, contrast, cast, crispness, noise, tilt |

## How it works

```
photo ─┬─ whole frame, true aspect ratio ── CLIP ViT-L/14 ─┐
       ├─ 4 full-resolution crops ───────── CLIP ViT-B/16 ─┼─ shared tower ── 20 attribute scores
       └─ pixel measurements (28 numbers) ─────────────────┘
```

Both photos go through the *same* tower (that is the "twin"). The verdict is
`sigmoid(scale * (score_A - score_B))`, so swapping the photos flips the answer
exactly, and every verdict breaks down into per-attribute gaps.

Training labels are human judgements only:

| Data | Size | Teaches |
|---|---|---|
| AVA | ~78k photo-contest images, ~200 votes each | aesthetic appeal |
| AADB | 10k photos, 11 rated attributes | balance, thirds, subject emphasis, light, colour harmony, depth of field, ... |
| SPAQ | 11k phone photos, 5 rated attributes | technical quality, exposure, contrast, sharpness, noise |
| KonIQ-10k | 10k photos | technical quality |
| Synthetic pairs | clean original vs. one known fault | exposure, white balance, level horizon, framing |

Held-out accuracy for every head is in `models/report.json` (written by training).
Heads that predict poorly are still displayed but never used for advice.

## Rebuild the model

```bash
python -m photocoach.fetch <url> data/raw/<file>     # resumable downloader (see extract.py for the files)
python -m photocoach.extract aadb koniq pros spaq ava   # cache features (GPU, ~2 h on a 1080 Ti)
python -m photocoach.train                              # trains the tower in ~2 min
```

## Layout

```
photocoach/
  backbone.py   frozen encoders and the image views fed to them
  signals.py    pixel measurements (exposure, clipping, colour, focus, noise, tilt, placement)
  model.py      the twin network
  synth.py      single-fault degradations with known ground truth
  extract.py    feature caching        train.py   training + held-out report
  objects.py    panoptic segmentation, subject selection, per-object removal, mask geometry
  scene.py      same-scene matching, framing comparison, look-matching edit, EXIF
  coach.py      score gaps + measurements -> explanation and advice
  engine.py     end-to-end comparison and overlays
  app.py        web app (static/index.html)
legacy/         the previous attempt, untouched
```

## Limits

- Object detection knows about 130 everyday categories (people, animals, vehicles, boats, buildings, sky, water, trees...). Unusual subjects are not outlined, and objects under about 0.2% of the frame are ignored.
- A "distraction" is a small object whose removal raises the aesthetic score by 3+ points. The removal is a quick paint-out, so treat small effects as hints. Large regions are only blurred, which shows how much the score depends on them, not whether the photo is better without them.

- It judges general appeal, not intent: a deliberately dark, soft or tilted photo is scored as a mistake.
- Scores within ~10 points are not meaningfully different.
- Light, timing and subject cannot be edited in; the edit preview tells you how much of the gap is that.
