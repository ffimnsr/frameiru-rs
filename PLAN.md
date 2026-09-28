# Frameiru Implementation Plan & Technical Specification

Specification plan derived from [HL.md](file:///home/pastel/Projects/frameiru/HL.md) for building a high-performance, real-time Linux virtual webcam with background removal/blur in Rust, designed as a modular engine supporting CLI, background daemon, and **Slint-based GUI**.

---

## 1. System Architecture Overview

```text
+-------------------------+      +-------------------------+      +-------------------------+
|      v4l2 Source        |      |      ONNX Runtime       |      |     v4l2loopback Sink   |
|      /dev/video0        |      |   (MODNet / RMBG-2.0)   |      |      /dev/video10       |
+------------+------------+      +------------+------------+      +------------+------------+
             |                                |                                ^
             v                                v                                |
+-------------------------+      +-------------------------+      +------------+------------+
|    frameiru-capture     | ---> |    frameiru-segment     | ---> |    frameiru-compose     |
| (V4L2 reader / Mock)    |      | (Letterbox + EMA Mask)  |      | (WGPU / CPU Blending)   |
+-------------------------+      +-------------------------+      +------------+------------+
             ^                                                                 |
             |                                                                 v
             +----------------------- frameiru-pipeline -----------------------+
                             (Decoupled Workers, Frame Pacing)
                                    |                   |
                        (Loopback Sink)       (Preview Broadcast Channel)
                                    |                   |
                                    v                   v
+-------------------+     +-------------------+   +--------------------+   +-------------------+
|   frameiru-cli    |     |  frameiru-daemon  |   |    frameiru-ui     |   | D-Bus / Tray App  |
| (Terminal UX)     |     | (IPC / Service)   |   | (Slint Desktop UI) |   | (System Shell)    |
+-------------------+     +-------------------+   +--------------------+   +-------------------+
```

### Primary Goals
1. **Low Latency & High FPS**: Target 30-60 FPS at 720p/1080p with end-to-end latency under 50ms.
2. **Decoupled Pipeline**: Real webcam capture and virtual webcam output run at native camera rate even if segmentation model runs slower (latest mask reuse).
3. **Zero-Allocation Hot Path**: Buffer pooling to avoid heap allocations in frame loops.
4. **Dual Backend**: High-performance GPU path (`wgpu`) with automated fallback to CPU blending.
5. **Runtime Dynamic Control**: Hot-reload background mode (blur, image, color) without restarting camera stream.
6. **GUI / Slint Ready Engine**:
   - Clean Rust `PipelineHandle` and `Engine` API that runs headlessly or embeds in a GUI event loop.
   - Zero-copy preview frame stream compatible with Slint's `SharedPixelBuffer<Rgb8Pixel>`.
   - Standalone preview mode (can run and test GUI without requiring `v4l2loopback` kernel module).
7. **Modular Crates**: Separation of concerns across independent, headless-friendly crates.

---

## 2. Multi-Crate Workspace Layout

```text
frameiru/
├── Cargo.toml                      # Workspace manifest
├── HL.md                           # High-level architecture doc
├── PLAN.md                         # This specification & implementation plan
├── AGENTS.md                       # Agent & coding standards
├── crates/
│   ├── frameiru-core/              # Common primitives, pixel formats, buffer pools, traits
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── buffer.rs           # BufferPool & FrameBuffer (zero-copy pooling)
│   │       ├── error.rs            # Central FrameiruError
│   │       ├── format.rs           # PixelFormat, Resolution, FrameMetadata
│   │       ├── lib.rs
│   │       ├── slint_compat.rs     # Optional conversion to slint::SharedPixelBuffer
│   │       └── traits.rs           # FrameSource, Segmenter, Compositor, FrameSink
│   ├── frameiru-capture/          # V4L2 webcam capture & synthetic mock generator
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── mock.rs             # Headless synthetic test frame generator
│   │       └── v4l2.rs             # Linux v4l2 capture implementation (mmap)
│   ├── frameiru-segment/          # ONNX inference & temporal mask smoothing
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── model.rs            # ONNX session wrapper (ort 2.x)
│   │       ├── preprocess.rs       # Aspect-ratio letterbox padding & normalization
│   │       ├── postprocess.rs      # Unletterbox crop & bilinear upsampling
│   │       └── smoother.rs         # Exponential Moving Average (EMA) filter
│   ├── frameiru-compose/          # Background replacement, blur, and rendering
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── cpu.rs              # CPU SIMD/parallel compositor fallback (rayon)
│   │       ├── gpu.rs              # WGPU compute/fragment shader compositor
│   │       ├── lib.rs
│   │       ├── mode.rs             # Blur, Image, SolidColor, Passthrough
│   │       └── shaders/            # WGSL shaders for compositing & dual kawase blur
│   ├── frameiru-sink/             # v4l2loopback output & broadcast sink
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── broadcast.rs        # In-memory broadcast sink for UI previews
│   │       ├── convert.rs          # Fast RGB <-> YUYV422 / NV12 (BT.601 / BT.709)
│   │       ├── lib.rs
│   │       ├── mock.rs             # Null sink for headless benchmarking
│   │       └── v4l2.rs             # v4l2loopback virtual device writer
│   ├── frameiru-ipc/              # Client/Server IPC protocol & D-Bus service
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── dbus.rs             # Optional D-Bus interface (zbus)
│   │       ├── lib.rs
│   │       ├── protocol.rs         # Commands: SetBackground, GetStatus, Stop
│   │       └── socket.rs           # Unix Domain Socket server & client
│   ├── frameiru-pipeline/         # Engine core, threading, channels, preview tap
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── config.rs           # PipelineConfig, DeviceConfig, ModelConfig
│   │       ├── engine.rs           # Embeddable Engine and PipelineHandle API
│   │       ├── lib.rs
│   │       ├── metrics.rs          # FPS, latency, frame drop tracker
│   │       └── runner.rs           # Multi-threaded pipeline coordinator
│   ├── frameiru-cli/              # CLI executable
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── commands/           # run, start, stop, status, set-bg, devices, bench
│   │       └── main.rs             # CLI entrypoint (clap)
│   └── frameiru-ui/               # Slint Desktop GUI application
│       ├── Cargo.toml              # Depends on frameiru-pipeline and slint
│       ├── ui/
│       │   └── main.slint          # Slint UI markup (webcam preview, sliders, toggles)
│       └── src/
│           ├── app.rs              # Slint event loop & property bindings
│           └── main.rs             # GUI launcher
└── models/                         # Default ONNX models or download scripts
```

---

## 3. Core Types and Trait Specifications

### 3.1 `frameiru-core`

#### Core Structs & Enums
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    Rgb8,
    Bgr8,
    Yuyv422,
    Nv12,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resolution {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BackgroundMode {
    Passthrough,
    Blur { radius: f32 },
    Color { r: u8, g: u8, b: u8 },
    Image { path: std::path::PathBuf },
}

#[derive(Debug, Clone)]
pub struct FrameMetadata {
    pub sequence: u64,
    pub timestamp_us: u64,
    pub resolution: Resolution,
    pub format: PixelFormat,
}

pub struct FrameBuffer {
    pub metadata: FrameMetadata,
    pub data: Vec<u8>,
}

pub struct Mask {
    pub resolution: Resolution,
    pub data: Vec<f32>, // Normalized 0.0 (background) to 1.0 (foreground)
}
```

#### Core Traits
```rust
pub trait FrameSource: Send + 'static {
    fn resolution(&self) -> Resolution;
    fn format(&self) -> PixelFormat;
    fn next_frame(&mut self) -> Result<FrameBuffer, FrameiruError>;
}

pub trait Segmenter: Send + 'static {
    fn input_resolution(&self) -> Resolution;
    fn segment(&mut self, frame: &FrameBuffer) -> Result<Mask, FrameiruError>;
}

pub trait Compositor: Send + 'static {
    fn composite(
        &mut self,
        source: &FrameBuffer,
        mask: &Mask,
        output: &mut FrameBuffer,
    ) -> Result<(), FrameiruError>;
    fn update_background(&mut self, mode: BackgroundMode) -> Result<(), FrameiruError>;
}

pub trait FrameSink: Send + 'static {
    fn write_frame(&mut self, frame: &FrameBuffer) -> Result<(), FrameiruError>;
}
```

---

## 4. Slint UI Integration Architecture

### 4.1 Embedding the Pipeline in Slint
Slint runs its own GUI event loop on the main OS thread (`slint::run_event_loop()`), while video capture, AI segmentation, and GPU compositing run in asynchronous background worker threads.

```text
+-----------------------------------------------------------+
| Main GUI Thread (Slint)                                   |
|   - main.slint UI markup (controls, video canvas)         |
|   - Properties: blur_radius, active_bg, fps_label         |
|   - slint::invoke_from_event_loop updates preview image   |
+-----------------------------+-----------------------------+
                              |
               (Command Calls | Preview Frame Stream)
                              v
+-----------------------------------------------------------+
| frameiru-pipeline::PipelineHandle                         |
|   - handle.set_background(BackgroundMode)                 |
|   - handle.subscribe_preview() -> broadcast::Receiver     |
|   - handle.stop() / handle.status()                       |
+-----------------------------------------------------------+
```

### 4.2 Slint Zero-Copy Frame Adapter
To render frames inside Slint with minimal copying:
```rust
// frameiru-core/src/slint_compat.rs (feature = "slint")
#[cfg(feature = "slint")]
use slint::{Image, Rgb8Pixel, SharedPixelBuffer};

#[cfg(feature = "slint")]
pub fn frame_to_slint_image(frame: &FrameBuffer) -> Option<Image> {
    if frame.metadata.format != PixelFormat::Rgb8 {
        return None;
    }
    let pixel_buffer = SharedPixelBuffer::<Rgb8Pixel>::clone_from_slice(
        bytemuck::cast_slice(&frame.data),
        frame.metadata.resolution.width,
        frame.metadata.resolution.height,
    );
    Some(Image::from_rgb8(pixel_buffer))
}
```

### 4.3 UI Preview Tap
In `frameiru-pipeline`:
- Compositor produces final `FrameBuffer` (RGB8).
- Output is sent to:
  1. `v4l2loopback` sink (if enabled).
  2. Preview `tokio::sync::broadcast::Sender<Arc<FrameBuffer>>` channel (ring capacity: 2, drop-oldest).
- Slint UI background task awaits preview frames and notifies UI:
```rust
let preview_rx = pipeline_handle.subscribe_preview();
let ui_weak = app.as_weak();

tokio::spawn(async move {
    let mut rx = preview_rx;
    while let Ok(frame) = rx.recv().await {
        if let Some(slint_image) = frame_to_slint_image(&frame) {
            let _ = ui_weak.upgrade_in_event_loop(move |ui| {
                ui.set_preview_frame(slint_image);
            });
        }
    }
});
```

### 4.4 Headless vs GUI Decoupling
- `frameiru-core`, `frameiru-pipeline`, `frameiru-segment`, `frameiru-compose`, and `frameiru-sink` have **zero GUI dependencies**.
- `frameiru-ui` is an independent crate depending on `slint` and `frameiru-pipeline`.
- Headless servers or daemon users install only CLI/daemon binaries without pulling `slint`, `winit`, or fontconfig.

---

## 5. Subsystem Details & Technical Decisions

### 5.1 Capture (`frameiru-capture`)
- **Backend**: `v4l` crate with `Memory::Mmap` stream for zero-copy kernel transfers.
- **Negotiation**: Query device capabilities, negotiate native resolution (e.g. 1280x720) and format (preferred: MJPEG or YUYV).
- **Format Decoding**:
  - MJPEG: fast JPEG decoder (`zune-jpeg` or `turbojpeg`).
  - YUYV: fast SIMD convert to RGB24.
- **Device Disconnect Recovery**: Gracefully handle `ENODEV` / `EIO` without crashing; emit test card pattern while attempting reconnect.
- **Mock Source**: Synthetic procedural frame generator (moving shapes / gradient) for CI test suites and GUI testing without physical webcams.

### 5.2 Segmentation (`frameiru-segment`)
- **Engine**: `ort` (ONNX Runtime 2.x bindings).
- **Supported Models**:
  - **RMBG-2.0** (BiRefNet architecture, high edge accuracy).
  - **MODNet** (Matting-oriented, low latency, 512x512).
  - **MediaPipe Selfie** (Lightweight, 256x256, ultra-fast for low-power CPUs).
- **Execution Providers**:
  - CPU (default with OpenMP / parallel threads).
  - CUDA / TensorRT (optional feature flag for NVIDIA).
  - DirectML / ROCm / CoreML / Vulkan where applicable.
- **Aspect Ratio Letterboxing**:
  - Webcams (16:9 / 4:3) -> Model input (square 512x512 or 256x256).
  - Preprocessing scales image with uniform aspect ratio and pads borders (letterbox).
  - Postprocessing unpads mask before upscaling to camera frame resolution to prevent human distortion.
- **Temporal Mask Smoother**:
  $$M_t = \alpha \cdot M_{\text{raw}} + (1 - \alpha) \cdot M_{t-1}$$
  Where $\alpha \in [0.4, 0.8]$ eliminates edge flickering while preventing trailing ghost artifacts.

### 5.3 Compositor (`frameiru-compose`)
- **Modes**:
  - `Blur { radius: f32 }`: Dual Kawase blur or Box blur applied to background pixels.
  - `Image { path: PathBuf }`: Static replacement image loaded as texture.
  - `Color { r: u8, g: u8, b: u8 }`: Solid color (green screen for OBS or clean backdrop).
  - `Passthrough`: Zero-cost bypass.
- **GPU Path (`wgpu`)**:
  - Offscreen rendering pipeline via compute or fragment shaders.
  - Passes:
    1. Upload camera frame & mask to GPU textures.
    2. Optional background blur pass (downsample -> dual kawase passes -> upsample).
    3. Alpha blend: $C_{\text{out}} = C_{\text{fg}} \cdot \text{mask} + C_{\text{bg}} \cdot (1 - \text{mask})$.
    4. Read back or render to target texture.
- **CPU Fallback**:
  - Multithreaded chunk blending via `rayon`.
  - Linear blending with fixed-point math / SIMD for fast CPU computation.

### 5.4 Sink (`frameiru-sink`)
- **Backend**: Writes to `/dev/videoX` created by `v4l2loopback`.
- **Target Pixel Format**: V4L2 loopback virtual devices standard is usually `V4L2_PIX_FMT_YUYV` or `V4L2_PIX_FMT_RGB24`.
  - Fast RGB24 to YUYV422 SIMD conversion function (ITU-R BT.601 color matrix).
- **Preview Broadcast Sink**: Broadcast channel publisher for Slint UI or IPC monitor clients.
- **Format Handshake**: Set format, width, height, and FPS on virtual device prior to streaming.
- **Mock Sink**: Discards or verifies frame integrity for headless stress tests and GUI-only mode.

### 5.5 IPC & Control Layer (`frameiru-ipc`)
- **Unix Domain Socket (UDS)**:
  - Socket path: `$XDG_RUNTIME_DIR/frameiru/control.sock`.
  - Protocol: Length-delimited JSON commands:
    - `{"cmd": "SetBackground", "mode": {"Blur": {"radius": 15.0}}}`
    - `{"cmd": "GetStatus"}`
    - `{"cmd": "Stop"}`
- **D-Bus Integration (Optional Feature `dbus`)**:
  - Service: `org.frameiru.Camera`.
  - Path: `/org/frameiru/Camera`.
  - Allows desktop shell / system tray extensions (GNOME Shell, KDE Plasma) to toggle background blur slider or virtual background.

### 5.6 Pipeline & Threading Model (`frameiru-pipeline`)
- To prevent inference stuttering the webcam stream:
  - **Thread 1 (Capture)**: Captures frames at 30/60 FPS, sends latest frame to `InferenceWorker` channel (capacity 1, drop-oldest) and sends full frame to `CompositorWorker`.
  - **Thread 2 (Inference)**: Pulls latest frame, runs ONNX inference, applies temporal smoother, updates shared `ArcSwap<Mask>`.
  - **Thread 3 (Compositor & Sink)**: Takes camera frame + latest available mask, composites, converts format, and writes to `v4l2loopback` and/or broadcast channel.
  - **Thread 4 (Control / IPC)**: Listens for runtime configuration updates from Slint UI/CLI/D-Bus and atomically updates pipeline state.

```text
Capture Thread ----(Frame)-------------------------> Compositor Thread ---> Sink (v4l2loopback)
       |                                                    |
   (drop-oldest)                                            +-------------> Preview Broadcast (Slint UI)
       v                                                    |
Inference Thread ---> [Smooth Mask ArcSwap] ----------------+
                               ^
Control / UI Handle -----------+ (Dynamic Blur / Image Updates)
```

---

## 6. Linux Environment Requirements

### Kernel Module: `v4l2loopback`
Setup virtual webcam device:
```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="Frameiru Virtual Cam" exclusive_caps=1
```
*Note: `exclusive_caps=1` is required for Chromium, Google Chrome, Zoom, and Firefox to detect the device as a capture source rather than an output.*

### Systemd User Unit (`~/.config/systemd/user/frameiru.service`)
```ini
[Unit]
Description=Frameiru Virtual Webcam Service
After=default.target

[Service]
ExecStart=/usr/local/bin/frameiru daemon
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
```

---

## 7. CLI & GUI Application Design

### CLI Commands (`frameiru-cli`)
```text
frameiru [OPTIONS] <COMMAND>

Commands:
  run          Start virtual webcam pipeline in foreground
  start        Launch background daemon service
  stop         Stop running background daemon service
  status       Check status of running virtual camera and active background
  set-bg       Dynamically update background mode on running camera
  devices      List available V4L2 capture and loopback devices
  models       Manage / download ONNX segmentation models
  benchmark    Run inference and composition benchmark without video sink
  inspect      Inspect a camera device format and supported resolutions
```

### Slint GUI Application (`frameiru-ui`)
- Visual webcam preview window (live 30 FPS video canvas).
- Interactive background controls:
  - Blur slider (dynamic radius 1-30).
  - Background image file picker (PNG, JPG).
  - Solid color picker (Hex / RGB).
  - Passthrough toggle.
- Camera device dropdown selector & resolution/FPS selector.
- Model selector (RMBG-2.0, MODNet, MediaPipe) with download status indicators.
- Virtual webcam status indicator (green badge when `/dev/video10` is active).

---

## 8. Phased Implementation Roadmap

### Phase 1: Workspace Scaffolding & Core Primitives
- [x] Initialize Cargo workspace with all crate directories.
- [x] Implement `frameiru-core`:
  - [x] `PixelFormat`, `Resolution`, `FrameMetadata`, `FrameBuffer`, `BackgroundMode`.
  - [x] Zero-allocation `BufferPool`.
  - [x] Optional Slint image conversion helper (`slint_compat.rs`).
  - [x] Traits: `FrameSource`, `Segmenter`, `Compositor`, `FrameSink`.
  - [x] Unit tests for resolution helpers, buffer allocation, and pixel format calculations.

### Phase 2: Capture & Sink Roundtrip (Passthrough)
- [x] Implement `frameiru-capture`:
  - [x] V4L2 device streaming via `v4l` crate.
  - [x] Format decoding (YUYV to RGB24, MJPEG to RGB24).
  - [x] Synthetic `MockSource` generating animated test pattern for tests.
- [x] Implement `frameiru-sink`:
  - [x] V4L2 virtual device loopback writer.
  - [x] In-memory `BroadcastSink` for preview subscribers.
  - [x] RGB24 to YUYV422 SIMD conversion.
  - [x] `MockSink` for validation.
- [x] Test: End-to-end webcam passthrough (`/dev/video0` -> passthrough -> `/dev/video10`).

### Phase 3: ONNX Segmentation Engine & Letterboxing
- [x] Implement `frameiru-segment`:
  - [x] Integration with `ort` crate.
  - [x] Aspect-ratio letterbox preprocessing and normalization.
  - [x] Model runner (MODNet, RMBG-2.0, MediaPipe).
  - [x] Postprocessing: unletterbox and bilinear upsample to frame resolution.
  - [x] `TemporalSmoother` with configurable $\alpha$.
- [x] Model download helper for standard ONNX weights.
- [x] Unit tests for preprocessing tensor shapes, smoother exponential formula, and bilinear upscaler.

### Phase 4: Compositor
- [x] Implement `frameiru-compose`:
  - [x] `cpu.rs`: Multi-threaded CPU fallback compositor using `rayon` (blur, replace image, color).
  - [x] `gpu.rs`: `wgpu` rendering pipeline (texture upload, WGSL shader blending, blur pass).
  - [x] Background modes: `Blur`, `Image`, `Color`, `Passthrough`.
- [x] Automated fallback logic: try WGPU; if initialization fails, log warning and use CPU compositor.

### Phase 5: Pipeline Engine & Preview Stream
- [x] Implement `frameiru-pipeline`:
  - [x] Embeddable `Engine` and cloneable `PipelineHandle`.
  - [x] Channel architecture separating capture, inference, and composition.
  - [x] Dynamic mask slot (`ArcSwap` or crossbeam channel with drop policy).
  - [x] Preview stream via `broadcast::Sender<Arc<FrameBuffer>>` for UI consumers.
  - [x] Frame rate pacing loop with latency and FPS counters.
  - [x] Clean shutdown handler.

### Phase 6: IPC & Dynamic Control
- [x] Implement `frameiru-ipc`:
  - [x] Unix Domain Socket server and client.
  - [x] IPC message protocol for status, background changes, and shutdown.
  - [x] Optional `zbus` D-Bus provider.

### Phase 7: CLI Application
- [x] Implement `frameiru-cli`:
  - [x] `clap` parser with subcommands (`run`, `start`, `stop`, `status`, `set-bg`, `devices`, `benchmark`, `models`).
  - [x] Auto-discovery for physical webcams and loopback devices.
  - [x] Standalone daemon management.

### Phase 8: Slint GUI Application (`frameiru-ui`)
- [ ] Implement `main.slint`:
  - Live video preview widget (`Image` component).
  - Sliders for blur radius, buttons for background selection, device selector dropdown.
- [ ] Implement `frameiru-ui/src/app.rs`:
  - Connect Slint event handlers to `PipelineHandle`.
  - Stream preview frames into `slint::SharedPixelBuffer` and update UI image on event loop.
  - Support headless preview mode without loopback module.

### Phase 9: Verification, Hardening & Benchmarks
- [ ] Quiet test suite (`cargo test --quiet`).
- [ ] Clippy checks with zero warnings.
- [ ] Flaky test protection: isolated synthetic sources and mocked sinks, no global shared state.
- [ ] End-to-end verification with virtual camera and browser / video consumer.
