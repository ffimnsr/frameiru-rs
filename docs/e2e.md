# End-to-end verification (Phase 9)

The full hardware loop — physical camera → segmentation → compositing →
virtual camera → video consumer — needs real devices. This page documents
the exact procedure; the automated parts are wired into the test suite and
skip cleanly when the hardware is absent.

## Prerequisites

```sh
# Virtual camera device (/dev/video10). exclusive_caps makes it show up as
# a camera (not a capture device) for consumers.
sudo modprobe v4l2loopback exclusive_caps=1 card_label="Frameiru"
# A physical camera at /dev/video0.
```

## 1. Automated roundtrip test

`crates/frameiru-sink/tests/passthrough.rs` runs the loop in-process:

```sh
cargo test -p frameiru-sink --features v4l2 --test passthrough -- --nocapture
```

Expected: `roundtrip ok: <W>x<H> via /dev/video10`. Skips with a message
when `/dev/video0` or `/dev/video10` is missing. Override devices with
`FRAMEIRU_VIDEO_DEVICE` / `FRAMEIRU_LOOPBACK_DEVICE`.

What it does: `V4l2Source(/dev/video0)` → `LoopbackSink(/dev/video10)` →
read back `/dev/video10` as a capture source → assert resolution/format.

## 2. CLI live pipeline

```sh
cargo build --release -p frameiru-cli --features full

# No model needed: the MediaPipe selfie-landscape model is embedded in the
# binary (256x144, ~2.9 ms/mask measured). Just run:
frameiru-cli run --background blur:8 --socket /tmp/frameiru.sock

# Or as a daemon:
frameiru-cli start
frameiru-cli status
frameiru-cli set-bg color:0,120,0
frameiru-cli stop

# External ONNX models still work via --model (must match input-size and
# normalization):
frameiru-cli run --model /path/to/rvm_mobilenetv3_fp32.onnx \
    --input-size 256x256 --background blur:8
```

The embedded default is MediaPipe Selfie Segmentation. External models default to imagenet normalization; RVM (u2net family) needs `--input-size` matching the graph (256x256 for RVM,
320x320 for silueta/u2net, 1024x1024 for isnet/BiRefNet/rmbg-2.0).

## 3. Video consumer check

With the pipeline running, verify the virtual camera with any consumer:

```sh
# Browser: open meet.google.com / zoom / a webcam test page, pick
# "Frameiru" as the camera.

# CLI consumers:
ffplay -f v4l2 -input_format yuyv422 -video_size 640x480 -i /dev/video10
ffmpeg -f v4l2 -i /dev/video10 -frames:v 10 out.mp4
```

The preview should show the composited output: subject kept, background
blurred/replaced/colored, and the foreground mask stable (no flicker).

## 4. GUI

```sh
cargo run -p frameiru-ui -- --model models/silueta.onnx --input-size 320x320
```

Start the pipeline from the window, toggle background modes, verify the
preview updates live and `stop`/window close shuts everything down cleanly.

## What was verified without hardware

- Full test matrix (all feature combinations), clippy zero-warning, three
  consecutive quiet test runs.
- GUI launches and runs (verified on a live display; visual check pending).
- Engine/CLI/IPC/GUI compose in-process (mock source + null sink) and are
  covered by unit/integration tests.

If you run the hardware steps above and hit an issue, the most useful
diagnostics are:

```sh
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video10 --list-formats-ext
journalctl -k | grep -i v4l2
```
