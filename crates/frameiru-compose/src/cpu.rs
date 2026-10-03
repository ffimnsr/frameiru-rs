//! Multi-threaded CPU compositor (rayon).
//!
//! Blends `source` with the resolved background using the mask:
//! `out = mask * fg + (1 - mask) * bg`. Blur mode applies a separable
//! sliding-window box blur to the source first; image mode stretches the
//! background via bilinear sampling. All row/column work is parallelized
//! with rayon.

use frameiru_core::buffer::Mask;
use frameiru_core::error::FrameiruError;
use frameiru_core::format::PixelFormat;
use frameiru_core::traits::Compositor;
use frameiru_core::{BackgroundMode, FrameBuffer, OverlayMode, Resolution};
use rayon::prelude::*;

use crate::mode::Background;

/// CPU fallback compositor.
pub struct CpuCompositor {
    background: Background,
    /// Full-frame overlay applied after the background blend.
    overlay: OverlayMode,
    /// Subject fill light in [0, 1]; 0 disables (see
    /// [`Compositor::set_subject_light`]).
    subject_light: f32,
    /// Horizontal blur pass scratch (one allocation for the compositor's life).
    scratch: Vec<u8>,
}

impl CpuCompositor {
    pub fn new() -> Self {
        Self {
            background: Background::Passthrough,
            overlay: OverlayMode::None,
            subject_light: 0.0,
            scratch: Vec::new(),
        }
    }

    pub fn background(&self) -> &Background {
        &self.background
    }
}

impl Default for CpuCompositor {
    fn default() -> Self {
        Self::new()
    }
}

impl Compositor for CpuCompositor {
    fn update_background(&mut self, mode: BackgroundMode) -> Result<(), FrameiruError> {
        self.background = Background::resolve(mode)?;
        Ok(())
    }

    fn update_overlay(&mut self, overlay: OverlayMode) {
        self.overlay = overlay;
    }

    fn set_subject_light(&mut self, light: f32) {
        self.subject_light = light.clamp(0.0, 1.0);
    }

    fn composite(
        &mut self,
        source: &FrameBuffer,
        mask: &Mask,
        output: &mut FrameBuffer,
    ) -> Result<(), FrameiruError> {
        if source.metadata.format != PixelFormat::Rgb8 {
            return Err(FrameiruError::FormatMismatch {
                expected: PixelFormat::Rgb8,
                actual: source.metadata.format,
            });
        }
        let res = source.metadata.resolution;
        if !res.is_valid() {
            return Err(FrameiruError::InvalidArgument(
                "source resolution must be non-zero".into(),
            ));
        }
        if mask.resolution != res || mask.data.len() < res.area() as usize {
            return Err(FrameiruError::InvalidArgument(format!(
                "mask resolution {:?} does not match source resolution {:?}",
                mask.resolution, res
            )));
        }

        let (w, h) = (res.width as usize, res.height as usize);
        let area = w * h;
        output.metadata = source.metadata;
        output.data.resize(area * 3, 0);

        match &self.background {
            Background::Passthrough => {
                output.data.copy_from_slice(&source.data[..area * 3]);
            }
            Background::Blur { radius } => {
                if self.scratch.len() < area * 3 {
                    self.scratch.resize(area * 3, 0);
                }
                blur_rgb8_masked(
                    &source.data[..area * 3],
                    mask,
                    w,
                    h,
                    *radius,
                    &mut self.scratch,
                    &mut output.data,
                );
                blend(
                    &self.scratch,
                    &source.data,
                    mask,
                    w,
                    self.subject_light,
                    &mut output.data,
                );
            }
            Background::Color { r, g, b } => {
                blend_color(
                    &source.data,
                    mask,
                    w,
                    self.subject_light,
                    *r,
                    *g,
                    *b,
                    &mut output.data,
                );
            }
            Background::Image { data, resolution } => {
                blend_image(
                    &source.data,
                    mask,
                    w,
                    h,
                    self.subject_light,
                    data,
                    *resolution,
                    &mut output.data,
                );
            }
            Background::Video { source: video } => {
                let frame = video.current();
                blend_image(
                    &source.data,
                    mask,
                    w,
                    h,
                    self.subject_light,
                    &frame.data,
                    frame.resolution,
                    &mut output.data,
                );
            }
        }

        apply_overlay(
            &mut output.data,
            w,
            h,
            self.overlay,
            source.metadata.timestamp_us,
        );
        Ok(())
    }
}

/// Applies the full-frame overlay pass on top of the finished composite.
/// All effects are per-pixel multipliers/additions; the animation phase is
/// driven by the source frame's timestamp so it runs even when the scene is
/// static.
fn apply_overlay(out: &mut [u8], w: usize, h: usize, mode: OverlayMode, timestamp_us: u64) {
    let t = (timestamp_us % (3600 * 1_000_000)) as f64 / 1_000_000.0;
    match mode {
        OverlayMode::None => {}
        OverlayMode::Scanlines => {
            out.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
                let factor = if y % 4 >= 2 { 0.50 } else { 1.0 };
                for c in row {
                    *c = ((*c as f32) * factor) as u8;
                }
            });
        }
        OverlayMode::LightLeak => {
            let wf = w as f64;
            let hf = h as f64;
            // Warm blob drifting on a slow Lissajous path.
            let cx = (0.5 + 0.35 * (t * 0.7).sin()) * wf;
            let cy = (0.25 + 0.2 * (t * 0.9 + 1.3).sin()) * hf;
            let r2 = (wf.max(hf) * 0.65).powi(2);
            out.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
                let dy = y as f64 - cy;
                let dy2 = dy * dy;
                for x in 0..w {
                    let dx = x as f64 - cx;
                    let d2 = dx * dx + dy2;
                    if d2 >= r2 {
                        continue;
                    }
                    let blob = 1.0 - d2 / r2;
                    let k = (blob * blob * 0.35) as f32;
                    let i = x * 3;
                    row[i] = (row[i] as f32 + 255.0 * k).min(255.0) as u8;
                    row[i + 1] = (row[i + 1] as f32 + 178.0 * k).min(255.0) as u8;
                    row[i + 2] = (row[i + 2] as f32 + 89.0 * k).min(255.0) as u8;
                }
            });
        }
        OverlayMode::Crt => {
            let wf = w as f64;
            let hf = h as f64;
            let flicker = 1.0 + 0.05 * (t * 15.0).sin();
            let roll_phase = t * 5.0;
            out.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
                let scan = if y % 4 >= 2 { 0.45 } else { 1.0 };
                let y_norm = (y as f64 + 0.5) / hf;
                let roll = 1.0 + 0.10 * (y_norm * std::f64::consts::PI * 4.0 - roll_phase).sin();
                let ry = y_norm - 0.5;
                let ry2 = ry * ry;
                for x in 0..w {
                    let rx = (x as f64 + 0.5) / wf - 0.5;
                    let vignette = (1.0 - 1.5 * (rx * rx + ry2)).clamp(0.0, 1.0);
                    let k = (scan * roll * vignette * flicker) as f32;
                    let sub = x % 3;
                    let rgb_triad = if sub == 0 {
                        (1.15f32, 0.88f32, 0.88f32)
                    } else if sub == 1 {
                        (0.88f32, 1.15f32, 0.88f32)
                    } else {
                        (0.88f32, 0.88f32, 1.15f32)
                    };
                    let i = x * 3;
                    row[i] = (row[i] as f32 * k * rgb_triad.0).min(255.0) as u8;
                    row[i + 1] = (row[i + 1] as f32 * k * rgb_triad.1).min(255.0) as u8;
                    row[i + 2] = (row[i + 2] as f32 * k * rgb_triad.2).min(255.0) as u8;
                }
            });
        }
    }
}

/// `out = mask * fg + (1 - mask) * bg` for a precomputed `bg` buffer, then a
/// masked fill-light lift: `+ light * mask * (255 - out)`.
fn blend(bg: &[u8], fg: &[u8], mask: &Mask, w: usize, light: f32, out: &mut [u8]) {
    out.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
        let base = y * w;
        for x in 0..w {
            let i = (base + x) * 3;
            let m = mask.data[base + x];
            let lm = light * m;
            for c in 0..3 {
                let f = fg[i + c] as f32;
                let b = bg[i + c] as f32;
                let v = m * f + (1.0 - m) * b;
                row[x * 3 + c] = (v + lm * (255.0 - v)).round() as u8;
            }
        }
    });
}

/// Solid-color variant of the blend (no background buffer needed).
#[allow(clippy::too_many_arguments)]
fn blend_color(fg: &[u8], mask: &Mask, w: usize, light: f32, r: u8, g: u8, b: u8, out: &mut [u8]) {
    let bg = [r as f32, g as f32, b as f32];
    out.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
        let base = y * w;
        for x in 0..w {
            let i = (base + x) * 3;
            let m = mask.data[base + x];
            let lm = light * m;
            for c in 0..3 {
                let f = fg[i + c] as f32;
                let v = m * f + (1.0 - m) * bg[c];
                row[x * 3 + c] = (v + lm * (255.0 - v)).round() as u8;
            }
        }
    });
}

/// Bilinear-sampled image background stretched to the frame.
#[allow(clippy::too_many_arguments)]
fn blend_image(
    fg: &[u8],
    mask: &Mask,
    w: usize,
    h: usize,
    light: f32,
    img: &[u8],
    img_res: Resolution,
    out: &mut [u8],
) {
    let (iw, ih) = (img_res.width as usize, img_res.height as usize);
    let sample = |sx: f32, sy: f32| -> [f32; 3] {
        let x0 = (sx.floor() as usize).min(iw - 1);
        let y0 = (sy.floor() as usize).min(ih - 1);
        let x1 = (x0 + 1).min(iw - 1);
        let y1 = (y0 + 1).min(ih - 1);
        let fx = sx - x0 as f32;
        let fy = sy - y0 as f32;
        let mut out = [0f32; 3];
        for c in 0..3 {
            let p00 = img[(y0 * iw + x0) * 3 + c] as f32;
            let p10 = img[(y0 * iw + x1) * 3 + c] as f32;
            let p01 = img[(y1 * iw + x0) * 3 + c] as f32;
            let p11 = img[(y1 * iw + x1) * 3 + c] as f32;
            let top = p00 + (p10 - p00) * fx;
            let bottom = p01 + (p11 - p01) * fx;
            out[c] = top + (bottom - top) * fy;
        }
        out
    };

    out.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
        let sy = (y as f32 + 0.5) * (ih as f32 / h as f32) - 0.5;
        let sy = sy.clamp(0.0, (ih - 1) as f32);
        for x in 0..w {
            let sx = (x as f32 + 0.5) * (iw as f32 / w as f32) - 0.5;
            let sx = sx.clamp(0.0, (iw - 1) as f32);
            let bg = sample(sx, sy);
            let m = mask.data[y * w + x];
            // Light wrap (U9.5): spill the background color over the subject
            // edge — effective alpha `m*(1 - wrap)` where wrap = 0.5*(1 - m),
            // i.e. `0.5*m*(1 + m)`; at m == 0/1 this is the plain blend.
            let eff = 0.5 * m * (1.0 + m);
            let lm = light * m;
            for c in 0..3 {
                let f = fg[(y * w + x) * 3 + c] as f32;
                let v = eff * f + (1.0 - eff) * bg[c];
                row[x * 3 + c] = (v + lm * (255.0 - v)).round() as u8;
            }
        }
    });
}

/// Foreground-aware separable sliding-window blur (two passes).
///
/// Kernel samples are weighted by `1 - alpha` and foreground pixels
/// (alpha >= 0.5) are excluded entirely, so bright subject colors cannot
/// smear over the matte edge into the background — halo-free (U9.5). An
/// all-zero mask degenerates to the plain box blur. Fully foreground
/// windows fall back to the center sample (subject stays crisp).
fn blur_rgb8_masked(
    src: &[u8],
    mask: &Mask,
    w: usize,
    h: usize,
    radius: u32,
    scratch: &mut [u8],
    out: &mut [u8],
) {
    let r = radius as i64;
    // Exclude foreground from background blur: any pixel with alpha >= 0.2 is excluded
    // to strictly prevent subject / skin colors from bleeding into the background blur.
    let wgt = |alpha: f32| {
        if alpha >= 0.20 {
            0.0
        } else {
            1.0 - (alpha / 0.20)
        }
    };

    // Horizontal pass: sliding weighted window over each row, writing into `out`.
    out.par_chunks_mut(w * 3)
        .enumerate()
        .for_each(|(y, row_out)| {
            let row_in = &src[y * w * 3..(y + 1) * w * 3];
            let mut lo: i64 = 0;
            let mut hi: i64 = -1;
            let mut sum = [0f32; 3];
            let mut wsum = 0f32;
            for x in 0..w {
                let hi_t = (x as i64 + r).min(w as i64 - 1);
                while hi < hi_t {
                    hi += 1;
                    let wi = wgt(mask.data[y * w + hi as usize]);
                    wsum += wi;
                    for c in 0..3 {
                        sum[c] += wi * row_in[(hi * 3 + c as i64) as usize] as f32;
                    }
                }
                let lo_t = (x as i64 - r).max(0);
                while lo < lo_t {
                    let wi = wgt(mask.data[y * w + lo as usize]);
                    wsum -= wi;
                    for c in 0..3 {
                        sum[c] -= wi * row_in[(lo * 3 + c as i64) as usize] as f32;
                    }
                    lo += 1;
                }
                if wsum > 0.0 {
                    for c in 0..3 {
                        row_out[x * 3 + c] = (sum[c] / wsum).round() as u8;
                    }
                } else {
                    row_out[x * 3..x * 3 + 3].copy_from_slice(&row_in[x * 3..x * 3 + 3]);
                }
            }
        });

    // Vertical pass: scan horizontally blurred rows from `out`, writing the full 2D
    // blurred background into `scratch` so `blend(&self.scratch, ...)` receives it.
    scratch
        .par_chunks_mut(w * 3)
        .enumerate()
        .for_each(|(y, row_scratch)| {
            let lo = (y as i64 - r).max(0);
            let hi = (y as i64 + r).min(h as i64 - 1);
            for x in 0..w {
                let mut sum = [0f32; 3];
                let mut wsum = 0f32;
                for ry in lo..=hi {
                    let wi = wgt(mask.data[ry as usize * w + x]);
                    wsum += wi;
                    let base = ((ry as usize) * w + x) * 3;
                    for c in 0..3 {
                        sum[c] += wi * out[base + c] as f32;
                    }
                }
                if wsum > 0.0 {
                    for c in 0..3 {
                        row_scratch[x * 3 + c] = (sum[c] / wsum).round() as u8;
                    }
                } else {
                    row_scratch[x * 3..x * 3 + 3]
                        .copy_from_slice(&out[(y * w + x) * 3..(y * w + x) * 3 + 3]);
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use frameiru_core::format::FrameMetadata;

    fn res(w: u32, h: u32) -> Resolution {
        Resolution {
            width: w,
            height: h,
        }
    }

    fn frame(res: Resolution, fill: u8) -> FrameBuffer {
        FrameBuffer::new(FrameMetadata {
            sequence: 0,
            timestamp_us: 0,
            resolution: res,
            format: PixelFormat::Rgb8,
        })
        .tap_fill(fill)
    }

    trait TapFill {
        fn tap_fill(self, fill: u8) -> Self;
    }
    impl TapFill for FrameBuffer {
        fn tap_fill(mut self, fill: u8) -> Self {
            self.data.fill(fill);
            self
        }
    }

    fn mask(res: Resolution, fill: f32) -> Mask {
        Mask {
            resolution: res,
            data: vec![fill; res.area() as usize],
        }
    }

    fn frame_of(res: Resolution, px: &[u8]) -> FrameBuffer {
        FrameBuffer::new(FrameMetadata {
            sequence: 0,
            timestamp_us: 0,
            resolution: res,
            format: PixelFormat::Rgb8,
        })
        .tap_pixels(px)
    }

    trait TapPixels {
        fn tap_pixels(self, px: &[u8]) -> Self;
    }
    impl TapPixels for FrameBuffer {
        fn tap_pixels(mut self, px: &[u8]) -> Self {
            self.data.copy_from_slice(px);
            self
        }
    }

    #[test]
    fn passthrough_copies_source() {
        let mut c = CpuCompositor::new();
        c.update_background(BackgroundMode::Passthrough).unwrap();
        let src = frame(res(4, 4), 200);
        let m = mask(res(4, 4), 0.0);
        let mut out = FrameBuffer::new(FrameMetadata {
            sequence: 0,
            timestamp_us: 0,
            resolution: res(4, 4),
            format: PixelFormat::Rgb8,
        });
        c.composite(&src, &m, &mut out).unwrap();
        assert_eq!(out.data, src.data);
    }

    #[test]
    fn blend_formula_with_solid_color() {
        let mut c = CpuCompositor::new();
        c.update_background(BackgroundMode::Color { r: 0, g: 0, b: 0 })
            .unwrap();
        let src = frame(res(3, 1), 200); // fg = 200 everywhere
        let mut out = FrameBuffer::new(FrameMetadata {
            sequence: 0,
            timestamp_us: 0,
            resolution: res(3, 1),
            format: PixelFormat::Rgb8,
        });

        // mask 1.0 -> fg; mask 0.0 -> bg; mask 0.5 -> exact midpoint.
        let m = Mask {
            resolution: res(3, 1),
            data: vec![1.0, 0.0, 0.5],
        };
        c.composite(&src, &m, &mut out).unwrap();
        assert_eq!(out.data, [200, 200, 200, 0, 0, 0, 100, 100, 100]);
    }

    #[test]
    fn subject_light_lifts_only_foreground() {
        let mut c = CpuCompositor::new();
        c.set_subject_light(0.5);
        c.update_background(BackgroundMode::Color { r: 0, g: 0, b: 0 })
            .unwrap();
        let src = frame(res(3, 1), 128);
        let m = Mask {
            resolution: res(3, 1),
            data: vec![1.0, 0.0, 0.5],
        };
        let mut out = FrameBuffer::new(FrameMetadata {
            sequence: 0,
            timestamp_us: 0,
            resolution: res(3, 1),
            format: PixelFormat::Rgb8,
        });
        c.composite(&src, &m, &mut out).unwrap();
        // mask 1: v = 128 + 0.5*1*(255-128) = 191.5 -> 192 (lit).
        assert_eq!(out.data[0], 192, "subject lifted toward white");
        // mask 0: pure background, untouched.
        assert_eq!(out.data[3], 0);
        // mask 0.5: v = 64 + 0.5*0.5*(255-64) = 111.75 -> 112.
        assert_eq!(out.data[6], 112, "half-mask lit halfway");
    }

    #[test]
    fn subject_light_zero_matches_plain_blend() {
        let mut c = CpuCompositor::new(); // default: no light
        c.update_background(BackgroundMode::Color {
            r: 10,
            g: 20,
            b: 30,
        })
        .unwrap();
        let src = frame(res(2, 1), 200);
        let m = Mask {
            resolution: res(2, 1),
            data: vec![0.8, 0.2],
        };
        let mut out = FrameBuffer::new(FrameMetadata {
            sequence: 0,
            timestamp_us: 0,
            resolution: res(2, 1),
            format: PixelFormat::Rgb8,
        });
        c.composite(&src, &m, &mut out).unwrap();
        // r: 0.8*200 + 0.2*10 = 162; b: 0.8*200 + 0.2*30 = 166.
        assert_eq!(out.data[0], 162);
        assert_eq!(out.data[2], 166);
        assert_eq!(out.data[3], 48); // 0.2*200 + 0.8*10 = 48
    }

    #[test]
    fn blur_preserves_uniform_field() {
        let mut c = CpuCompositor::new();
        c.update_background(BackgroundMode::Blur { radius: 3.0 })
            .unwrap();
        let src = frame(res(16, 16), 100);
        let m = mask(res(16, 16), 0.0); // fully background
        let mut out = FrameBuffer::new(FrameMetadata {
            sequence: 0,
            timestamp_us: 0,
            resolution: res(16, 16),
            format: PixelFormat::Rgb8,
        });
        c.composite(&src, &m, &mut out).unwrap();
        assert!(out.data.iter().all(|&v| v == 100));
    }

    #[test]
    fn blur_smooths_vertical_step_edge() {
        // Top half white, bottom half black: vertical blur must smooth the horizontal boundary.
        let w = 8;
        let h = 16;
        let mut px = vec![0u8; w * h * 3];
        for row in px[..w * (h / 2) * 3].iter_mut() {
            *row = 255;
        }
        let src = frame_of(res(w as u32, h as u32), &px);
        let mut c = CpuCompositor::new();
        c.update_background(BackgroundMode::Blur { radius: 3.0 })
            .unwrap();
        let m = mask(res(w as u32, h as u32), 0.0);
        let mut out = FrameBuffer::new(FrameMetadata {
            sequence: 0,
            timestamp_us: 0,
            resolution: res(w as u32, h as u32),
            format: PixelFormat::Rgb8,
        });
        c.composite(&src, &m, &mut out).unwrap();
        let mid = out.data[w * (h / 2) * 3];
        assert!((mid as i32 - 128).abs() <= 40, "mid = {mid}");
    }

    #[test]
    fn blur_smooths_step_edge() {
        // Left half white, right half black: after blur the border is a ramp.
        let w = 32;
        let mut px = vec![255u8; w * 4 * 3];
        for row in px.chunks_exact_mut(w * 3) {
            for v in row[w / 2 * 3..].iter_mut() {
                *v = 0;
            }
        }
        let src = frame_of(res(w as u32, 4), &px);
        let mut c = CpuCompositor::new();
        c.update_background(BackgroundMode::Blur { radius: 4.0 })
            .unwrap();
        let m = mask(res(w as u32, 4), 0.0);
        let mut out = FrameBuffer::new(FrameMetadata {
            sequence: 0,
            timestamp_us: 0,
            resolution: res(w as u32, 4),
            format: PixelFormat::Rgb8,
        });
        c.composite(&src, &m, &mut out).unwrap();
        // Midpoint column should be near 128, and strictly ramping.
        let mid = out.data[w / 2 * 3];
        assert!((mid as i32 - 128).abs() <= 32, "mid = {mid}");
        let left = out.data[0];
        let right = out.data[(w - 1) * 3];
        assert!(left as u16 > mid as u16 + 16, "left {left} vs mid {mid}");
        assert!(mid as u16 > right as u16 + 16, "mid {mid} vs right {right}");
    }

    /// U9.5 regression: with a sharp subject (alpha 1) on the left of a
    /// bright/dark step, the blurred background right of the edge must stay
    /// pure dark — the old full-window box blur smeared the bright subject
    /// across the matte edge (halo).
    #[test]
    fn blur_excludes_foreground_from_kernel() {
        let w = 32; // left 16 px subject (255), right 16 px background (0)
        let mut px = vec![255u8; w * 4 * 3];
        for row in px.chunks_exact_mut(w * 3) {
            for v in row[w / 2 * 3..].iter_mut() {
                *v = 0;
            }
        }
        let src = frame_of(res(w as u32, 4), &px);
        let mut m = mask(res(w as u32, 4), 0.0);
        for y in 0..4usize {
            for x in 0..(w / 2) {
                m.data[y * w + x] = 1.0;
            }
        }
        let mut c = CpuCompositor::new();
        c.update_background(BackgroundMode::Blur { radius: 4.0 })
            .unwrap();
        let mut out = FrameBuffer::new(FrameMetadata {
            sequence: 0,
            timestamp_us: 0,
            resolution: res(w as u32, 4),
            format: PixelFormat::Rgb8,
        });
        c.composite(&src, &m, &mut out).unwrap();
        // Background side: no bright anything, even adjacent to the edge.
        for y in 0..4usize {
            for x in (w / 2)..w {
                let i = (y * w + x) * 3;
                assert_eq!(&out.data[i..i + 3], &[0, 0, 0], "halo at row {y} col {x}");
            }
        }
        // Subject side (alpha 1) composites the source unchanged.
        for y in 0..4usize {
            for x in 0..(w / 2) {
                let i = (y * w + x) * 3;
                assert_eq!(
                    &out.data[i..i + 3],
                    &[255, 255, 255],
                    "subject pixel (row {y}, col {x})"
                );
            }
        }
    }

    /// U9.5: image backgrounds spill their color over the subject edge.
    /// At alpha 0.5 with a white fg over a blue bg, the effective alpha
    /// drops to 0.375: the red channel is pulled from 128 (plain blend)
    /// toward the blue background; alpha 0/1 stay exact.
    #[test]
    fn light_wrap_spills_background_on_subject_edge() {
        let mut c = CpuCompositor::new();
        c.update_background(BackgroundMode::Image {
            path: temp_image(0, 0, 255).into(),
        })
        .unwrap();
        let src = frame(res(3, 1), 255);
        let m = Mask {
            resolution: res(3, 1),
            data: vec![1.0, 0.5, 0.0],
        };
        let mut out = FrameBuffer::new(FrameMetadata {
            sequence: 0,
            timestamp_us: 0,
            resolution: res(3, 1),
            format: PixelFormat::Rgb8,
        });
        c.composite(&src, &m, &mut out).unwrap();
        // Alpha 1: subject untouched.
        assert_eq!(&out.data[..3], &[255, 255, 255]);
        // Alpha 0.5: 0.375 * fg + 0.625 * bg per channel.
        assert_eq!(out.data[3], (0.375f32 * 255.0).round() as u8, "red wrapped");
        assert_eq!(out.data[5], 255, "blue stays background blue");
        // Alpha 0: pure background.
        assert_eq!(&out.data[6..9], &[0, 0, 255]);
    }

    #[test]
    fn image_background_stretches_to_frame() {
        // 1x1 red image stretched over a 2x2 frame, mask 0 -> all red.
        let mut c = CpuCompositor::new();
        c.update_background(BackgroundMode::Image {
            path: temp_image(255, 0, 0).into(),
        })
        .unwrap();
        let src = frame(res(2, 2), 0);
        let m = mask(res(2, 2), 0.0);
        let mut out = FrameBuffer::new(FrameMetadata {
            sequence: 0,
            timestamp_us: 0,
            resolution: res(2, 2),
            format: PixelFormat::Rgb8,
        });
        c.composite(&src, &m, &mut out).unwrap();
        assert!(out.data.chunks(3).all(|p| p == [255, 0, 0]));
    }

    #[test]
    fn rejects_mismatched_mask_resolution() {
        let mut c = CpuCompositor::new();
        let src = frame(res(4, 4), 0);
        let m = mask(res(2, 2), 0.5);
        let mut out = FrameBuffer::new(FrameMetadata {
            sequence: 0,
            timestamp_us: 0,
            resolution: res(4, 4),
            format: PixelFormat::Rgb8,
        });
        assert!(c.composite(&src, &m, &mut out).is_err());
    }

    #[test]
    fn rejects_non_rgb_source() {
        let mut c = CpuCompositor::new();
        let src = FrameBuffer::new(FrameMetadata {
            sequence: 0,
            timestamp_us: 0,
            resolution: res(2, 2),
            format: PixelFormat::Yuyv422,
        });
        let m = mask(res(2, 2), 0.5);
        let mut out = FrameBuffer::new(FrameMetadata {
            sequence: 0,
            timestamp_us: 0,
            resolution: res(2, 2),
            format: PixelFormat::Rgb8,
        });
        assert!(c.composite(&src, &m, &mut out).is_err());
    }

    #[test]
    fn overlay_none_is_identity() {
        let mut c = CpuCompositor::new();
        c.update_background(BackgroundMode::Passthrough).unwrap();
        c.update_overlay(OverlayMode::None);
        let src = frame(res(4, 4), 210);
        let m = mask(res(4, 4), 0.0);
        let mut out = FrameBuffer::new(FrameMetadata {
            sequence: 0,
            timestamp_us: 0,
            resolution: res(4, 4),
            format: PixelFormat::Rgb8,
        });
        c.composite(&src, &m, &mut out).unwrap();
        assert_eq!(out.data, src.data);
    }

    #[test]
    fn scanlines_darken_odd_rows_only() {
        let mut c = CpuCompositor::new();
        c.update_background(BackgroundMode::Color { r: 0, g: 0, b: 0 })
            .unwrap();
        c.update_overlay(OverlayMode::Scanlines);
        let src = frame(res(4, 4), 200);
        let m = mask(res(4, 4), 1.0); // fully foreground: blend keeps 200
        let mut out = FrameBuffer::new(FrameMetadata {
            sequence: 0,
            timestamp_us: 0,
            resolution: res(4, 4),
            format: PixelFormat::Rgb8,
        });
        c.composite(&src, &m, &mut out).unwrap();
        for y in 0..4 {
            let expected = if y % 4 >= 2 {
                (200.0f32 * 0.50).round() as u8
            } else {
                200
            };
            assert_eq!(
                &out.data[y * 4 * 3..y * 4 * 3 + 3],
                &[expected; 3],
                "row {y}"
            );
        }
    }

    #[test]
    fn light_leak_lifts_warm_patch_only() {
        let mut c = CpuCompositor::new();
        c.update_background(BackgroundMode::Color { r: 0, g: 0, b: 0 })
            .unwrap();
        c.update_overlay(OverlayMode::LightLeak);
        let mut src = frame(res(64, 48), 0);
        // t = 5.0s puts the blob near (24, 8); the bottom-right corner must
        // stay outside its radius.
        src.metadata.timestamp_us = 5_000_000;
        let m = mask(res(64, 48), 0.0);
        let mut out = FrameBuffer::new(FrameMetadata {
            sequence: 0,
            timestamp_us: 5_000_000,
            resolution: res(64, 48),
            format: PixelFormat::Rgb8,
        });
        c.composite(&src, &m, &mut out).unwrap();
        // Blob center: warm tint added (255, 178, 89) * 0.35.
        let center = out.data[(8 * 64 + 24) * 3];
        assert!(
            (85..=93).contains(&center),
            "red at blob center was {center}, want ~89"
        );
        // Corner outside the radius stays pure black.
        assert_eq!(
            &out.data[(47 * 64 + 63) * 3..(47 * 64 + 63) * 3 + 3],
            &[0, 0, 0]
        );
    }

    #[test]
    fn crt_vignettes_and_darkens_odd_rows() {
        let mut c = CpuCompositor::new();
        c.update_background(BackgroundMode::Color { r: 0, g: 0, b: 0 })
            .unwrap();
        c.update_overlay(OverlayMode::Crt);
        let src = frame(res(32, 32), 255);
        let m = mask(res(32, 32), 1.0);
        let mut out = FrameBuffer::new(FrameMetadata {
            sequence: 0,
            timestamp_us: 0, // t = 0 -> flicker factor exactly 1.0
            resolution: res(32, 32),
            format: PixelFormat::Rgb8,
        });
        c.composite(&src, &m, &mut out).unwrap();

        let center = out.data[(16 * 32 + 16) * 3];
        assert!(
            center > 220,
            "center pixel must be ~untouched at t=0, got {center}"
        );
        let corner = out.data[0];
        assert!(corner < 200, "corner must be vignetted, got {corner}");
        // Odd rows get the extra scanline darkening on top of the vignette.
        let bright_row = out.data[(16 * 32 + 16) * 3];
        let scan_row = out.data[(18 * 32 + 16) * 3];
        assert!(scan_row < bright_row, "scanline row darker than bright row at same column");
    }

    /// Writes a solid-color PNG to a temp file and returns its path.
    fn temp_image(r: u8, g: u8, b: u8) -> String {
        let path = std::env::temp_dir().join(format!(
            "frameiru-test-{}-{r}-{g}-{b}.png",
            std::process::id()
        ));
        let img = image::RgbImage::from_pixel(1, 1, image::Rgb([r, g, b]));
        img.save(&path).unwrap();
        path.to_string_lossy().into_owned()
    }
}
