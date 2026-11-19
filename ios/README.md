# Photo Coach for iOS (on-device)

Everything runs on the phone: no server, and photos never leave the device.

## Build

Needs a Mac with Xcode 15 or later and an iPhone on iOS 17+.

```
brew install xcodegen
cd ios
xcodegen            # writes PhotoCoach.xcodeproj from project.yml
open PhotoCoach.xcodeproj
```

In Xcode: select the PhotoCoach target, set your Team under Signing & Capabilities, change the
bundle identifier from `com.example.photocoach` to one of your own, then run on a device.
(The simulator works too, but Core ML is much slower there.)

## First thing to do: run the self-test

Tap **Self-test** (top right). It runs two bundled photos through the on-device pipeline and
compares the results with what the PyTorch model produced for the same files. The Core ML
models were converted on Linux, where they cannot be executed, so this screen is the actual
verification that conversion and Swift preprocessing match training. Attribute percentiles
within about 3 points of the reference mean the port is faithful.

## What is in the app

| File | Role |
|---|---|
| `Resources/Encoder.mlpackage` | image encoder: 256x256 image -> 512-d embedding, run 5x per photo (whole frame + 4 detail crops) |
| `Resources/Tower.mlpackage` | the twin tower: embeddings + measured signals -> 20 attribute scores |
| `Resources/judge.json` | attribute names, calibration tables, comparison scales, reliability flags, advice text |
| `Resources/parity.json`, `parity_*.jpg` | reference outputs for the self-test |
| `Sources/Judge/ImagePrep.swift` | decode, orient, standard resize, squash + crops (mirrors the Python preprocessing) |
| `Sources/Judge/Signals.swift` | tone / colour measurements fed to the tower |
| `Sources/Judge/PhotoJudge.swift` | runs the models, scores a photo, compares two |
| `Sources/Judge/Coach.swift` | verdict, reasons, advice |
| `Sources/UI/*` | SwiftUI screens |

## Rebuilding the models

From the repo root, on the training machine:

```
PHOTOCOACH_DATA=/path/to/data python -m mobile.extract aadb koniq pros spaq ava
python -m mobile.train
python -m mobile.export_coreml      # rewrites ios/PhotoCoach/Resources
```

## Not on device yet

Compared with the desktop web app, this first build has the verdict, attribute breakdown,
reasons and per-attribute advice. Still to port: measured evidence in the advice (clipping,
focus map, EXIF), same-scene framing comparison, the "edit toward the winner" preview, and
object / distraction analysis (planned on Apple's Vision framework rather than Mask2Former).
