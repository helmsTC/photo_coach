# ApertureIQ - Photo Coach for Android (on-device)

A port of the iOS app in `../ios`. Everything runs on the phone: no server, and photos never leave the device.

## Build

Needs JDK 17 and the Android SDK (platform 34). Android Studio supplies both; open the `android` folder in it, or build from a terminal:

```
cd android
./gradlew :app:assembleDebug        # app/build/outputs/apk/debug/app-debug.apk
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Minimum Android version is 8.0 (API 26). The debug APK is about 280 MB: 183 MB of models, plus native libraries for four CPU types and unshrunk code. A release build for one CPU type is much smaller.

If the checkout sits on a slow or shared drive (for example a Windows drive under WSL), build outside the source tree:

```
APERTUREIQ_BUILD=~/.cache/apertureiq ./gradlew --project-cache-dir ~/.cache/apertureiq/.gradle :app:assembleDebug
```

## Tests: the port is checked against PyTorch on a desktop

`core` is plain Kotlin with no Android classes, and ONNX Runtime has the same Java API on a desktop JVM as on Android. So the whole analysis pipeline runs in unit tests:

```
./gradlew :core:test
```

- **ParityTest** scores the two bundled test photos and compares them with what the PyTorch model produced for the same files (`assets/parity.json`). All 20 attribute percentiles land within 1 point.
- **EndToEndTest** runs full comparisons on sample pairs: scene matching, objects, attention maps, edit previews and coaching. It also checks that swapping the photos flips the verdict exactly, and that the free and Pro tiers agree.

What the desktop tests cannot cover is the Android-only code: decoding photos with `BitmapFactory`, the Compose screens, billing and history.

## First thing to do on a phone: run the self-test

In a debug build, tap **⋯ > Run self-test** on the home screen. It runs the same parity check on the device, through Android's own image decoder. One of the two test photos carries a colour profile, so this is also the check that pixel values reach the network unconverted (see `platform/Engine.kt`). Results also go to logcat under the tag `ApertureIQ`.

The two headline scores must be within 3 percentile points of the reference; the other attributes are allowed 8. Android's JPEG decoder produces pixels a fraction of a level different from the reference decoder (PIL), and that alone moves the less stable attributes by up to about 6 points. Verified on a TCL NXTPAPER 11 Plus (Android 15): the PyTorch model given the tablet's decoded pixels reproduces the app's numbers exactly. Debug builds save those pixels as `files/selftest_<name>.rgb` for this check.

## What is in the app

| Path | Role |
|---|---|
| `core/.../judge/Imaging.kt` | pixel buffers, Gaussian blur, Lab conversion, the area resize that mirrors OpenCV, squash + crops |
| `core/.../judge/Signals.kt` | port of `signals.py`: tone, colour, focus map, noise, horizon, visual weight, palette, histogram |
| `core/.../judge/Scene.kt` | same-scene alignment, framing comparison, "match the look" recipe, crop-to-winner |
| `core/.../judge/Objects.kt` | objects from the segmenter, subject / distraction logic, paint-out for re-scoring |
| `core/.../judge/Overlays.kt` | clipping, focus, AI-attention and object overlays |
| `core/.../judge/Coach.kt` | port of `coach.py`: verdict, reasons, evidence-backed coaching |
| `core/.../judge/Authenticity.kt` | AI-image check: labels in the file, then the detector network |
| `core/.../judge/PhotoJudge.kt` | the staged comparison and the free tier's quick verdict |
| `core/.../judge/OrtNetworks.kt` | runs the four networks through ONNX Runtime |
| `app/.../platform/Engine.kt` | Android decoding (orientation, raw pixel values, EXIF) and model loading from the APK |
| `app/.../ui/*` | Compose screens: splash and tier choice, home, analysis progress, free verdict, Pro results (overview, inspector lenses, coaching plan, before/after edit, scores, AI check), paywall, history, about |
| `app/src/main/assets/` | models, `judge.json`, self-test fixtures and sample photos |

## Models

| File | Size | What |
|---|---|---|
| `encoder.onnx` | 88 MB | OpenAI CLIP ViT-B/32 image encoder (MIT) with 8-bit weights, 224 x 224 -> 512-d, run on 5 views per photo |
| `tower.onnx` | 6 MB | the twin tower: embeddings + measured signals -> 20 attribute scores |
| `segmenter.onnx` | 45 MB | DETR-ResNet50 panoptic (`facebook/detr-resnet-50-panoptic`, Apache-2.0), weights stored as 8-bit lookup tables, 448 x 448 -> one COCO class per pixel |
| `ai_detector.onnx` | 44 MB | Community Forensics ViT-S/16-384 (MIT), P(AI-generated) |

Weights are stored as 16-bit floats behind `Cast` nodes that ONNX Runtime folds back to 32-bit at load time, so the files are half size but inference is ordinary float32 on the CPU. The models are stored uncompressed in the APK and memory-mapped, so they are not copied to app storage.

To rebuild them, from the repo root on the training machine:

```
python -m mobile.export_onnx        # rewrites android/app/src/main/assets, checking each model against PyTorch
```

`judge.json`, the parity fixtures and the sample photos are copied from the iOS resources; run `python -m mobile.export_coreml` first if the judge was retrained.

## Differences from the iOS app

The analysis code is a line-for-line port, with these substitutions for Apple frameworks:

- **Networks:** ONNX Runtime on the CPU instead of Core ML. Expect comparisons to take longer than on an iPhone's Neural Engine. NNAPI or XNNPACK can be switched on through the `configure` hook of `OrtNetworks`, but neither has been tried, and the self-test should be re-run if you do.
- **Objects:** the same DETR panoptic model, exported from the original PyTorch checkpoint rather than Apple's Core ML build. There is no equivalent of Vision's person-instance masks, so a group of touching people stays one region (as in the iOS simulator).
- **Attention / saliency:** the desktop app's spectral-residual map instead of Vision's attention saliency. This feeds the Composition lens and subject picking, not the scores.
- **Alignment:** the zoom + offset search only. iOS adds a candidate from Vision's homographic registration, so strongly perspective-shifted pairs that iOS matches may be reported here as "similar" rather than "same scene".
- **Image decoding:** large photos are reduced inside the decoder by a power of two, then area-averaged to 2048 px before analysis.
- **History:** saved copies are JPEG (quality 92, long side 2048) instead of HEIC.
- **Haptics and blur:** Android haptic constants; glass cards are flat translucent fills rather than live blurs.

## Free and Pro tiers

Same as iOS: **Free** shows which shot wins (read, measure, score and AI-check stages only); **Pro** is the full analysis. Pro is a one-time in-app product sold through Google Play Billing, product ID `com.mosaiclens.apertureiq.pro` (`ui/Store.kt`).

- **Shipping:** create a one-time (non-consumable) in-app product with that ID in the Play Console and upload a signed build to at least the internal testing track. Play only returns products for an app it knows, installed from Play or by a licence tester.
- **Debug builds without Play:** when Play has no product to offer, the button reads "Unlock Pro (debug build)" and grants Pro locally. Release builds never do this.

## Measured on a device

TCL NXTPAPER 11 Plus (MediaTek MT8781, 2 fast + 6 slow cores, 8 GB, Android 15), debug build, Pro comparison of a sample pair with the app already open for a few seconds:

| | First device run | Now |
|---|---|---|
| Verdict on screen | 72 s (with everything else) | about 7 s |
| Everything finished | 72 s | 43-48 s |
| Native memory afterwards | 2.5 GB | 0.67 GB |

What changed, none of it altering results (the JVM tests give the same numbers):

- **Verdict first.** `PhotoJudge.compare` hands over a first result after scoring and the AI check; Inspect and Coach show a progress list until the rest arrives.
- **Models load while photos are being chosen** (`Engine.warm`): 5-9 s that used to be part of the first comparison.
- **Stages run side by side.** The networks keep the fast cores; measuring, scene matching, object painting and overlays run on a low-priority pool. Each network has its own lock, so the segmenter and the encoder can overlap. The eight per-photo measurements also run in parallel.
- **Runtime settings** (`Engine.kt`, with the measurements in a comment): one model thread per fast core, ONNX Runtime's memory pool off, no XNNPACK, no NNAPI.

Also measured and rejected: the encoder in 16-bit arithmetic (no faster on this chip). A debug build loads `files/override/<name>.onnx` in place of a bundled model, which is how that was tried.

What is left is mostly network time on the two fast cores: about 20 s of encoder (107-123 images at 0.2 s) and 11 s of segmenter when run alone.

Self-test passes on the device. Debug builds log `STAGE` and `TIME` lines (per stage, per step, per network) under the logcat tag `ApertureIQ`, and read `adb shell setprop debug.aiq.threads|xnn|nnapi|arena|cpuprio|warm <n>` so settings can be tried without rebuilding (`xnn` and `nnapi` are bit masks: 1 encoder, 2 segmenter, 4 detector).

## Debug launch options

Debug builds read intent extras, for screenshots and testing:

```
adb shell am start -n com.mosaiclens.apertureiq/.MainActivity --ez nosplash true --ez pro true
adb shell am start -n com.mosaiclens.apertureiq/.MainActivity --ei autosample 0 --es tab coach --ez expandall true
adb shell am start -n com.mosaiclens.apertureiq/.MainActivity --ei autosample 3 --es lens objects --ei photo 0
adb shell am start -n com.mosaiclens.apertureiq/.MainActivity --ez selftest true
```

`autosample N` runs sample pair N; `tab` is overview, inspect, coach, scores or ai; `lens` is photo, clipping, focus, composition, objects, attention or framing; `free` picks the free tier.

## Before a Play release

- **Signing and bundle:** add a release signing config and build an app bundle (`./gradlew :app:bundleRelease`). The release build shrinks code with R8 (`proguard-rules.pro` keeps ONNX Runtime and the serializers).
- **Size:** Play allows 200 MB of compressed download for the base module. The models are 183 MB (166 MB compressed); with the 64-bit ARM libraries and code the download is estimated at about 180-190 MB. Check the real figure in Play Console's app size report after uploading a bundle. The AI detector stays 16-bit because 8-bit weights changed its verdicts; if more room is needed, move it or the segmenter into an install-time asset pack (Play Asset Delivery).
- **Model licences:** the About screen credits each model. All four allow commercial use: CLIP ViT-B/32 (MIT), Community Forensics (MIT), DETR (Apache 2.0); the tower is our own. Verify each on its model card before shipping.
