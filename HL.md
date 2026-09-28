### High-level architecture (Rust virtual webcam with background removal)

```text
+-------------------+        +----------------------+        +--------------------+
|  Real Webcam      |        |  Background Model    |        |  Virtual Webcam    |
|  /dev/video0      |        |  (ONNX, MODNet/rmbg) |        |  /dev/video10      |
+---------+---------+        +----------+-----------+        +---------+----------+
          |                             |                              |
          v                             v                              v
+-------------------+        +----------------------+        +--------------------+
|  Capture Service  |  --->  |  Segmentation &      |  --->  |  Compositor &      |
|  (v4l, Rust)      |        |  Mask Processing     |        |  v4l2loopback Out  |
+-------------------+        +----------------------+        +--------------------+
                 ^                                          
                 |                                          
         +-------------------+                              
         |  Control Layer    |                              
         |  (CLI / DBus / UI)|                              
         +-------------------+
```

---

### 1. Processes and responsibilities

#### **A. Capture service**
- **Purpose:** Read frames from the real webcam.
- **Tech:**
  - `v4l` crate to open `/dev/video0`.
  - Converts frames to a standard format (e.g., RGB).
- **Output:** `FrameRGB { width, height, pixels }`.

#### **B. Segmentation & mask processing**
- **Purpose:** Compute foreground mask (person vs background).
- **Tech:**
  - `onnxruntime` (Rust bindings).
  - Model: **rmbg 2.0** (MobileNetV3) or **MODNet** in ONNX.
- **Steps:**
  - Downscale frame to `160×160` or `256×256`.
  - Run model → `MaskLowRes`.
  - Temporal smoothing: blend with previous mask.
  - Upscale to original size → `MaskFullRes`.

#### **C. Compositor & v4l2loopback output**
- **Purpose:** Combine foreground with new background and expose as virtual webcam.
- **Tech:**
  - GPU path: `wgpu` (preferred) or OpenGL (`glium`/`glow`).
  - CPU fallback: `image`/`opencv` for blending.
  - `v4l` writer to `/dev/video10` (v4l2loopback).
- **Steps:**
  - Load background (image, blur, solid color).
  - Shader/compositor:
    - `out = mask * foreground + (1 - mask) * background`.
  - Convert RGB → YUYV/MJPEG.
  - Write frames at stable 30 FPS using `tokio::time::interval`.

#### **D. Control layer (CLI / DBus / UI)**
- **Purpose:** User control and desktop integration.
- **Interfaces:**
  - CLI like ShellCam:
    - `rustcam virtual start --bg path/to/bg.png`
    - `rustcam virtual stop`
    - `rustcam status`
  - Optional DBus service:
    - GNOME/Plasma integration.
    - Toggle background blur/replacement from tray or settings.

---

### 2. Module layout (Rust crates)

- **`rustcam-core`**
  - `capture::WebcamCapture`
  - `segmentation::OnnxSegmenter`
  - `mask::TemporalSmoother`
  - `compositor::GpuCompositor` / `CpuCompositor`
  - `virtual_cam::LoopbackWriter`

- **`rustcam-cli`**
  - CLI commands (similar to ShellCam).
  - Starts/stops the pipeline.

- **`rustcam-daemon`** (optional)
  - Long-running service.
  - DBus API for desktop integration.

---

### 3. Data flow (step-by-step)

1. **Capture**
   - `WebcamCapture` reads frame from `/dev/video0`.
2. **Preprocess**
   - Downscale frame → `FrameSmall`.
3. **Segment**
   - `OnnxSegmenter` runs model → `MaskLowRes`.
4. **Smooth**
   - `TemporalSmoother` blends masks → `MaskLowResSmooth`.
5. **Upscale**
   - `MaskFullRes` aligned with original frame.
6. **Composite**
   - `GpuCompositor` blends foreground + background → `FrameComposite`.
7. **Convert & output**
   - `LoopbackWriter` converts and writes to `/dev/video10`.
8. **Apps use virtual webcam**
   - Zoom/Meet/Teams/Discord select `/dev/video10`.

---

### 4. How this differs from OBS

- **OBS:** monolithic, heavy, streaming-focused, GUI-first.
- **Your architecture:** modular, CLI/daemon-first, Rust-native, optimized for webcam enhancement and v4l2loopback.

If you want, next step I can sketch concrete Rust function signatures for `capture`, `segmentation`, `compositor`, and `virtual_cam` so you can start coding without guessing.
