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
use frameiru_core::{BackgroundMode, FrameBuffer, Resolution};
use rayon::prelude::*;

use crate::mode::Background;

/// CPU fallback compositor.
pub struct CpuCompositor {
    background: Background,
    /// Horizontal blur pass scratch (one allocation for the compositor's life).
    scratch: Vec<u8>,
}

impl CpuCompositor {
    pub fn new() -> Self {
        Self {
            background: Background::Passthrough,
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
                return Ok(());
            }
            Background::Blur { radius } => {
                if self.scratch.len() < area * 3 {
                    self.scratch.resize(area * 3, 0);
                }
                blur_rgb8(
                    &source.data[..area * 3],
                    w,
                    h,
                    *radius,
                    &mut self.scratch,
                    &mut output.data,
                );
                blend(&self.scratch, &source.data, mask, w, &mut output.data);
            }
            Background::Color { r, g, b } => {
                blend_color(&source.data, mask, w, *r, *g, *b, &mut output.data);
            }
            Background::Image { data, resolution } => {
                blend_image(
                    &source.data,
                    mask,
                    w,
                    h,
                    data,
                    *resolution,
                    &mut output.data,
                );
            }
        }
        Ok(())
    }
}

/// `out = mask * fg + (1 - mask) * bg` for a precomputed `bg` buffer.
fn blend(bg: &[u8], fg: &[u8], mask: &Mask, w: usize, out: &mut [u8]) {
    out.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
        let base = y * w;
        for x in 0..w {
            let i = (base + x) * 3;
            let m = mask.data[base + x];
            for c in 0..3 {
                let f = fg[i + c] as f32;
                let b = bg[i + c] as f32;
                row[x * 3 + c] = (m * f + (1.0 - m) * b).round() as u8;
            }
        }
    });
}

/// Solid-color variant of the blend (no background buffer needed).
fn blend_color(fg: &[u8], mask: &Mask, w: usize, r: u8, g: u8, b: u8, out: &mut [u8]) {
    let bg = [r as f32, g as f32, b as f32];
    out.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
        let base = y * w;
        for x in 0..w {
            let i = (base + x) * 3;
            let m = mask.data[base + x];
            for c in 0..3 {
                let f = fg[i + c] as f32;
                row[x * 3 + c] = (m * f + (1.0 - m) * bg[c]).round() as u8;
            }
        }
    });
}

/// Bilinear-sampled image background stretched to the frame.
fn blend_image(
    fg: &[u8],
    mask: &Mask,
    w: usize,
    h: usize,
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
            for c in 0..3 {
                let f = fg[(y * w + x) * 3 + c] as f32;
                row[x * 3 + c] = (m * f + (1.0 - m) * bg[c]).round() as u8;
            }
        }
    });
}

/// Separable sliding-window box blur (two passes).
fn blur_rgb8(src: &[u8], w: usize, h: usize, radius: u32, scratch: &mut [u8], out: &mut [u8]) {
    let r = radius as i64;
    let window = |lo: i64, hi: i64| (hi - lo + 1) as u32;

    // Horizontal pass: sliding window over each row, parallel over rows.
    scratch
        .par_chunks_mut(w * 3)
        .enumerate()
        .for_each(|(y, row_out)| {
            let row_in = &src[y * w * 3..(y + 1) * w * 3];
            let mut lo: i64 = 0;
            let mut hi: i64 = -1;
            let mut sum = [0u32; 3];
            for x in 0..w {
                let hi_t = (x as i64 + r).min(w as i64 - 1);
                while hi < hi_t {
                    hi += 1;
                    for c in 0..3 {
                        sum[c] += row_in[(hi * 3 + c as i64) as usize] as u32;
                    }
                }
                let lo_t = (x as i64 - r).max(0);
                while lo < lo_t {
                    for c in 0..3 {
                        sum[c] -= row_in[(lo * 3 + c as i64) as usize] as u32;
                    }
                    lo += 1;
                }
                let n = window(lo, hi);
                for c in 0..3 {
                    row_out[x * 3 + c] = (sum[c] / n) as u8;
                }
            }
        });

    // Vertical pass: each output row scans the scratch rows in its window,
    // parallel over rows.
    out.par_chunks_mut(w * 3)
        .enumerate()
        .for_each(|(y, row_out)| {
            for x in 0..w {
                let lo = (y as i64 - r).max(0);
                let hi = (y as i64 + r).min(h as i64 - 1);
                let n = window(lo, hi);
                let mut sum = [0u32; 3];
                for ry in lo..=hi {
                    let base = ((ry as usize) * w + x) * 3;
                    for c in 0..3 {
                        sum[c] += scratch[base + c] as u32;
                    }
                }
                for c in 0..3 {
                    row_out[x * 3 + c] = (sum[c] / n) as u8;
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
    fn blur_smooths_step_edge() {
        // Left half white, right half black: after blur the border is a ramp.
        let w = 32;
        let mut px = vec![255u8; w * 4 * 3];
        for v in px[w / 2 * 3..].iter_mut() {
            *v = 0;
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
