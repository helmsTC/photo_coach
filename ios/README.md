# ApertureIQ - Photo Coach for iOS (on-device)

Everything runs on the phone: no server, and photos never leave the device.

## Build

Needs a Mac with Xcode 15 or later and an iPhone on iOS 17+.

```
brew install xcodegen
cd ios
xcodegen            # writes PhotoCoach.xcodeproj from project.yml
open PhotoCoach.xcodeproj
```

`project.yml` carries the signing team and bundle identifier, so re-running `xcodegen` keeps them;
change `DEVELOPMENT_TEAM` / `PRODUCT_BUNDLE_IDENTIFIER` there to build under another account.

The simulator works too; there the Core ML models run on the CPU (the simulator's GPU path returns
all zeros) and Vision's person masks are unavailable, so touching people stay one region.

## First thing to do: run the self-test

Tap **⋯ > Run self-test** on the home screen. It runs two bundled photos through the on-device pipeline and
compares the results with what the PyTorch model produced for the same files. The Core ML
models were converted on Linux, where they cannot be executed, so this screen is the actual
verification that conversion and Swift preprocessing match training. Aesthetic / technical within
about 3 points of the reference mean the port is faithful; a few narrow attribute heads can differ
by 5-8 points because the encoder runs in 16-bit precision.

For a headless run, launch with `-diag`: it prints the self-test, a horizon-direction check, an
alignment check on a known crop, and full comparisons of the bundled sample pairs to the console,
then exits:

```
xcrun simctl launch --console-pty booted com.mosaiclens.apertureiq -diag
```

## What is in the app

| File | Role |
|---|---|
| `Resources/Encoder.mlpackage` | image encoder (CLIP ViT-B/32): 224x224 image -> 512-d embedding, run 5x per photo (whole frame + 4 detail crops) |
| `Resources/Tower.mlpackage` | the twin tower: embeddings + measured signals -> 20 attribute scores |
| `Resources/judge.json` | attribute names, calibration tables, comparison scales, reliability flags, advice text |
| `Resources/parity.json`, `parity_*.jpg` | reference outputs for the self-test |
| `Resources/Segmenter.mlpackage` | Apple's Core ML DETR-ResNet50 semantic segmenter (COCO panoptic, 133 classes; Apache-2.0, from huggingface.co/apple/coreml-detr-semantic-segmentation, 8-bit palettized) |
| `Resources/sample_*.jpg` | built-in sample pairs on the home screen |
| `Sources/Judge/ImagePrep.swift` | decode, orient, standard resize, squash + crops (mirrors the Python preprocessing; ICC profiles are not applied, as in PIL) |
| `Sources/Judge/Signals.swift` | port of `signals.py`: tone, colour, focus map, noise, horizon, visual weight, palette, histogram |
| `Sources/Judge/Exif.swift` | camera settings from the file |
| `Sources/Judge/Scene.swift` | same-scene alignment, framing comparison, "match the look" recipe, crop-to-winner |
| `Sources/Judge/Objects.swift` | objects from the COCO segmenter (+ Vision person masks), subject / distraction logic from `objects.py`, paint-out for re-scoring |
| `Sources/Judge/Overlays.swift` | clipping, focus, AI-attention and object overlays |
| `Sources/Judge/PhotoJudge.swift` | runs the models; staged comparison: score, align, re-score each object painted out, occlusion map, edit previews |
| `Sources/Judge/Coach.swift` | port of `coach.py`: verdict, reasons, evidence-backed coaching |
| `Sources/Judge/Diagnostics.swift` | sample pairs, self-test, `-diag` console run |
| `Sources/UI/*` | SwiftUI screens: home, analysis progress, results (inspector lenses, coaching plan, before/after edit, breakdown, details) |

## Rebuilding the models

From the repo root, on the training machine:

```
PHOTOCOACH_DATA=/path/to/data python -m mobile.extract aadb koniq pros spaq ava
python -m mobile.train
python -m mobile.export_coreml      # rewrites ios/PhotoCoach/Resources
```

## Differences from the desktop app

Everything from the desktop app is on device now: measured evidence (clipping, focus, noise,
EXIF), same-scene framing, the "edit toward the winner" previews, the AI-attention map and
object / distraction analysis. Where the desktop uses OpenCV or Mask2Former the phone uses:

- **Objects:** Apple's DETR semantic segmenter (same COCO panoptic classes as Mask2Former, run at
  448 x 448) instead of Mask2Former. It labels pixels by class but does not separate instances, so
  objects are connected regions of a class; Vision's person masks split groups of people on device.
  Buildings, bridges and tents are never called distractions.
- **Alignment:** a zoom + offset search scored by structure correlation, plus Vision's image
  registration, instead of SIFT + MAGSAC. It handles re-framings of the same scene well; strong
  perspective changes are left to the Vision candidate.
- **Horizon:** the same line test as the desktop (long, near-level, different on each side), with a
  gradient + Hough vote in place of Canny + HoughLinesP.
- **Attention / saliency:** Vision's attention saliency in place of the spectral-residual map.

## Free and Pro tiers

After the splash, the user picks **Free** (which shot wins, nothing else; only the read → measure → score stages run) or **Pro** (the full analysis). Pro is a one-time, non-consumable StoreKit 2 purchase, product ID `com.mosaiclens.apertureiq.pro` (`Sources/UI/Store.swift`).

- **Testing:** the scheme uses `ApertureIQ.storekit`, so purchases work in Xcode runs (simulator or device) without App Store Connect. Use Debug → StoreKit → Manage Transactions to refund or reset.
- **Shipping:** create a Non-Consumable in-app purchase with the same product ID in App Store Connect, and sign the Paid Apps agreement.
- **Debug flags:** `-pro` grants Pro and `-free` picks the free tier. Both are DEBUG-only, like `-nosplash` and `-autosample`.

## Recent comparisons

Free keeps the last 3 comparisons and Pro keeps 10 (`Sources/UI/History.swift`). The limit follows Pro *ownership*, so a Pro owner who picks Free doesn't lose history. Everything stays on the device in Application Support/History:

- **Stored per comparison:** both photos as JPEG copies at most 2048px on the long side (EXIF and colour profile kept, GPS stripped), small thumbnails, and the verdict in `index.json`. That's roughly 1–2 MB per comparison.
- **Reopening:** a saved comparison re-runs the analysis in the current tier, so a Free entry can be reopened as a full Pro breakdown. `-diag -historycheck` confirms that saved copies get the same verdict as the originals.
- **Duplicates:** comparing the same two files again updates the existing entry. Sample pairs aren't saved; the DEBUG flag `-savesamples` overrides that, and `-openrecent N` reopens entry N.

## AI-image check

Each shot gets an AI check (`Sources/Judge/Authenticity.swift`, `UI/AuthenticityView.swift`). Free shows a badge on each photo; Pro adds an "AI check" card with the evidence.

1. **Labels in the file decide when present.** These are Content Credentials (C2PA) and IPTC `DigitalSourceType` (`trainedAlgorithmicMedia` means generated, `compositeWithTrainedAlgorithmicMedia` means edited), Stable Diffusion/ComfyUI generation settings, and known generator names in metadata (for example DALL·E's "Made with OpenAI DALL-E"). Only metadata segments are scanned, never pixel data. C2PA signatures are not verified.
2. **Otherwise the detector network gives a likelihood:** `AIDetector.mlpackage` (42 MB, FP16) is the Community Forensics ViT-S/16-384 by Jeongsoo Park and Andrew Owens (CVPR 2025), MIT license © 2025 Jeongsoo Park. It's exported by `mobile/export_ai_detector.py`. Thresholds: ≥ 0.93 gives "Likely AI", ≤ 0.15 gives "No AI signs", and anything between is "Can't tell".

Test results with the app's own code (`-diag -aicheck <folder>`):
- **Real photos (71):** none called AI and 4 "can't tell". These are 22 of our own, 24 RAISE camera originals and 25 COCO/ImageNet/LAION images.
- **AI images (81):** 73 flagged, 5 "can't tell" and 3 missed. Generators include FLUX, Midjourney-style, Firefly, Ideogram, Stable Cascade, LCM and DALL·E.
- **Recompressed images:** in a PyTorch test at messaging-app quality (1080px, JPEG q75), detection falls from about 95% to 72%. False alarms don't rise.

## Release setup

- **App icon:** `Resources/Assets.xcassets/AppIcon.appiconset/AppIcon.png` is rendered from `AppIconArt` (in `UI/AboutView.swift`). To regenerate it after a design change, launch a Debug build with `-exporticon <absolute path to AppIcon.png>`.
- **Export compliance:** `PhotoCoach/Info.plist` sets `ITSAppUsesNonExemptEncryption = NO`. It is merged into the generated Info.plist.
- **Privacy manifest:** `Resources/PrivacyInfo.xcprivacy` declares no tracking, no data collected and no required-reason APIs. Update it if UserDefaults, file timestamps or similar APIs are ever added.
- **About screen:** opened from the home screen's ••• menu. It shows the version, the privacy note, Restore purchases and the model credits. The self-test menu item only exists in Debug builds.
- **Saved comparisons:** copies are HEIC at quality 0.8 (about 40% smaller than the old JPEG q0.95; JPEG entries from earlier builds still open). Each entry also stores both shots' AI-check results from the original files, because the re-encoded copies lose C2PA/XMP labels and about a quarter of AI images stop being flagged after re-encoding. History is excluded from iCloud and computer backups.
- **Licensing:** the encoder is OpenAI's CLIP ViT-B/32 image tower (MIT license, © 2021 OpenAI), which allows commercial use. It replaced MobileCLIP2-S0, whose weights are limited to non-commercial research. The Tower is trained by us on the encoder's features. Verify the license on the model card (huggingface.co/openai/clip-vit-base-patch32) before shipping.
