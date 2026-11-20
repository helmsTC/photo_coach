# ApertureIQ - Photo Coach for Android (on-device)

A port of the iOS app in `../ios`. Everything runs on the phone: no server, and photos never leave the device.

## Build

Needs JDK 17 and the Android SDK (platform 34). Android Studio supplies both; open the `android` folder in it, or build from a terminal:

```
cd android
./gradlew :app:assembleDebug        # app/build/outputs/apk/debug/app-debug.apk
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Minimum Android version is 8.0 (API 26). The debug APK is about 400 MB: 313 MB of models, plus native libraries for four CPU types and unshrunk code. A release build for one CPU type is much smaller.

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
| `encoder.onnx` | 176 MB | OpenAI CLIP ViT-B/32 image encoder (MIT), 224 x 224 -> 512-d, run on 5 views per photo |
| `tower.onnx` | 6 MB | the twin tower: embeddings + measured signals -> 20 attribute scores |
| `segmenter.onnx` | 87 MB | DETR-ResNet50 panoptic (`facebook/detr-resnet-50-panoptic`, Apache-2.0), 448 x 448 -> one COCO class per pixel |
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
- **Size:** the models alone are 313 MB, which is over Play's size limit for an app's base module (check the current figure in the Play Console help). Before a Play release, move the models into an install-time asset pack (Play Asset Delivery), or shrink the encoder: 8-bit weights halve it to about 88 MB but move scores by 1-2 percentile points on average.
- **Model licences:** the About screen credits each model. All four allow commercial use: CLIP ViT-B/32 (MIT), Community Forensics (MIT), DETR (Apache 2.0); the tower is our own. Verify each on its model card before shipping.
