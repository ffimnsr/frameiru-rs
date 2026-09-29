//! Letterbox preprocessing: aspect-ratio padding and RGB8 -> normalized CHW.
//!
//! Portrait segmentation models expect a fixed square input (e.g. 320x320 or
//! 256x256). `Letterbox` scales the source to fit inside that canvas and
//! centers it, padding the borders. `preprocess_rgb8` then resizes (bilinear,
//! edge-clamped) and normalizes into a CHW `f32` tensor in one pass.

use frameiru_core::error::FrameiruError;
use frameiru_core::format::Resolution;

/// Aspect-ratio-preserving scale/pad geometry between source and model input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Letterbox {
    pub src: Resolution,
    pub target: Resolution,
    /// Uniform scale applied to `src` so it fits inside `target`.
    pub scale: f32,
    /// Scaled source size (<= target), centered inside the canvas.
    pub inner: Resolution,
    /// Left/right padding (inner is centered: right pad == left pad or +1).
    pub pad_x: u32,
    /// Top/bottom padding.
    pub pad_y: u32,
}

impl Letterbox {
    /// Computes scale and padding for `src` fitted into `target`.
    pub fn compute(src: Resolution, target: Resolution) -> Result<Self, FrameiruError> {
        if !src.is_valid() || !target.is_valid() {
            return Err(FrameiruError::InvalidArgument(format!(
                "letterbox requires non-zero resolutions, got {src:?} -> {target:?}"
            )));
        }
        let scale =
            (target.width as f32 / src.width as f32).min(target.height as f32 / src.height as f32);
        let inner = Resolution {
            width: (src.width as f32 * scale).round().min(target.width as f32) as u32,
            height: (src.height as f32 * scale)
                .round()
                .min(target.height as f32) as u32,
        };
        let pad_x = (target.width - inner.width) / 2;
        let pad_y = (target.height - inner.height) / 2;
        Ok(Self {
            src,
            target,
            scale,
            inner,
            pad_x,
            pad_y,
        })
    }

    /// Scaled source region inside the canvas.
    pub fn inner_size(&self) -> Resolution {
        self.inner
    }

    /// Top-left offset of the scaled region inside the canvas.
    pub fn inner_offset(&self) -> (u32, u32) {
        (self.pad_x, self.pad_y)
    }

    /// Leftover padding amount is at most one pixel on each side.
    pub fn is_centered(&self) -> bool {
        (self.target.width - self.inner.width) / 2 == self.pad_x
            && (self.target.height - self.inner.height) / 2 == self.pad_y
    }
}

/// Per-channel mean/std for tensor normalization.
///
/// Values are normalized as `((v / 255) - mean) / std`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Normalization {
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for Normalization {
    /// ImageNet statistics used by MODNet/RMBG-style portrait models.
    fn default() -> Self {
        Self {
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        }
    }
}

impl Normalization {
    pub fn new(mean: [f32; 3], std: [f32; 3]) -> Result<Self, FrameiruError> {
        if std.iter().any(|&s| s <= 0.0 || !s.is_finite()) {
            return Err(FrameiruError::InvalidArgument(
                "normalization std must be positive and finite".into(),
            ));
        }
        Ok(Self { mean, std })
    }

    /// Plain `v / 255` scaling (range [0, 1]), used by MediaPipe-style models.
    pub fn unit() -> Self {
        Self {
            mean: [0.0; 3],
            std: [1.0; 3],
        }
    }
}

/// Normalizes one 0..255 sample.
pub fn normalize_pixel(v: u8, mean: f32, std: f32) -> f32 {
    (v as f32 / 255.0 - mean) / std
}

/// Resizes, letterboxes, and normalizes `src` into CHW float layout.
///
/// `out` must hold at least `3 * target.area()` floats and is overwritten:
/// padding regions become `0.0` (in normalized space) and the letterboxed
/// inner region is bilinear-resampled from `src` (SIMD-accelerated via
/// `fast_image_resize`) with edge clamping.
pub fn preprocess_rgb8(
    src: &[u8],
    src_res: Resolution,
    target: Resolution,
    normalization: Normalization,
    out: &mut [f32],
) -> Result<Letterbox, FrameiruError> {
    let letterbox = Letterbox::compute(src_res, target)?;
    let need_in = src_res.area() as usize * 3;
    if src.len() < need_in {
        return Err(FrameiruError::InsufficientCapacity {
            need: need_in,
            have: src.len(),
        });
    }
    let (tw, th) = (target.width as usize, target.height as usize);
    let (iw, ih) = (
        letterbox.inner.width as usize,
        letterbox.inner.height as usize,
    );
    let (ox, oy) = (letterbox.pad_x as usize, letterbox.pad_y as usize);
    let total = tw * th;
    let need_out = total * 3;
    if out.len() < need_out {
        return Err(FrameiruError::InsufficientCapacity {
            need: need_out,
            have: out.len(),
        });
    }

    out[..need_out].fill(0.0);

    let (sw, sh) = (src_res.width as usize, src_res.height as usize);

    // Inner-region RGB. Debug builds use the fused scalar sampler
    // (`fast_image_resize`'s SIMD paths collapse to slow scalar fallbacks,
    // ~20 ms/call here); release builds get the SIMD resize. The 1:1 case
    // skips resizing in both.
    let inner: Vec<u8> = if iw == sw && ih == sh {
        src.to_vec()
    } else {
        #[cfg(debug_assertions)]
        {
            let mut buf = Vec::with_capacity(iw * ih * 3);
            for y in 0..ih {
                for x in 0..iw {
                    let sx = sample_coord(x as f32, iw as f32, sw as f32);
                    let sy = sample_coord(y as f32, ih as f32, sh as f32);
                    let (r, g, b) = bilinear_sample_rgb8(src, sw, sh, sx, sy);
                    buf.extend_from_slice(&[r, g, b]);
                }
            }
            buf
        }
        #[cfg(not(debug_assertions))]
        {
            let src_view = fast_image_resize::images::ImageRef::new(
                src_res.width,
                src_res.height,
                src,
                fast_image_resize::PixelType::U8x3,
            )
            .map_err(|e| FrameiruError::InvalidArgument(format!("resize input view: {e}")))?;
            let mut resized = fast_image_resize::images::Image::new(
                iw as u32,
                ih as u32,
                fast_image_resize::PixelType::U8x3,
            );
            let mut resizer = fast_image_resize::Resizer::new();
            let options = fast_image_resize::ResizeOptions::new().resize_alg(
                fast_image_resize::ResizeAlg::Interpolation(
                    fast_image_resize::FilterType::Bilinear,
                ),
            );
            resizer
                .resize(&src_view, &mut resized, &options)
                .map_err(|e| FrameiruError::InvalidArgument(format!("resize failed: {e}")))?;
            resized.into_vec()
        }
    };

    // Normalize the inner region into the CHW planes around the padding.
    for y in 0..ih {
        for x in 0..iw {
            let i = (y * iw + x) * 3;
            let dst = (oy + y) * tw + ox + x;
            out[dst] = normalize_pixel(inner[i], normalization.mean[0], normalization.std[0]);
            out[total + dst] =
                normalize_pixel(inner[i + 1], normalization.mean[1], normalization.std[1]);
            out[2 * total + dst] =
                normalize_pixel(inner[i + 2], normalization.mean[2], normalization.std[2]);
        }
    }
    Ok(letterbox)
}

/// Maps a destination pixel to a source coordinate (texel-center convention),
/// clamped to the source edge. Debug-only: release uses the SIMD resize.
#[cfg(debug_assertions)]
fn sample_coord(dst: f32, dst_size: f32, src_size: f32) -> f32 {
    let s = (dst + 0.5) * (src_size / dst_size) - 0.5;
    s.clamp(0.0, src_size - 1.0)
}

/// Bilinear sample with edge clamping; `sx`/`sy` are clamped by the caller.
#[cfg(debug_assertions)]
pub(crate) fn bilinear_sample_rgb8(
    src: &[u8],
    sw: usize,
    sh: usize,
    sx: f32,
    sy: f32,
) -> (u8, u8, u8) {
    let x0 = sx.floor().min((sw - 1) as f32) as usize;
    let y0 = sy.floor().min((sh - 1) as f32) as usize;
    let x1 = (x0 + 1).min(sw - 1);
    let y1 = (y0 + 1).min(sh - 1);
    let fx = sx - x0 as f32;
    let fy = sy - y0 as f32;

    let sample = |x: usize, y: usize, c: usize| src[(y * sw + x) * 3 + c] as f32;
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let chan = |c: usize| {
        let top = lerp(sample(x0, y0, c), sample(x1, y0, c), fx);
        let bottom = lerp(sample(x0, y1, c), sample(x1, y1, c), fx);
        lerp(top, bottom, fy).round() as u8
    };

    (chan(0), chan(1), chan(2))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn res(w: u32, h: u32) -> Resolution {
        Resolution {
            width: w,
            height: h,
        }
    }

    #[test]
    fn letterbox_horizontal_source() {
        // 640x480 into 320x320: scale 0.5, inner 320x240, vertical padding.
        let lb = Letterbox::compute(res(640, 480), res(320, 320)).unwrap();
        assert_eq!(lb.scale, 0.5);
        assert_eq!(lb.inner, res(320, 240));
        assert_eq!((lb.pad_x, lb.pad_y), (0, 40));
        assert!(lb.is_centered());
    }

    #[test]
    fn letterbox_vertical_source() {
        let lb = Letterbox::compute(res(480, 640), res(320, 320)).unwrap();
        assert_eq!(lb.inner, res(240, 320));
        assert_eq!((lb.pad_x, lb.pad_y), (40, 0));
    }

    #[test]
    fn letterbox_same_aspect_is_fit_exact() {
        let lb = Letterbox::compute(res(640, 480), res(640, 480)).unwrap();
        assert_eq!(lb.scale, 1.0);
        assert_eq!(lb.inner, res(640, 480));
        assert_eq!((lb.pad_x, lb.pad_y), (0, 0));
    }

    #[test]
    fn letterbox_rejects_zero_dimensions() {
        assert!(Letterbox::compute(res(0, 10), res(10, 10)).is_err());
        assert!(Letterbox::compute(res(10, 10), res(0, 10)).is_err());
    }

    #[test]
    fn normalization_defaults_are_imagenet() {
        let n = Normalization::default();
        assert_eq!(n.mean, [0.485, 0.456, 0.406]);
        assert_eq!(n.std, [0.229, 0.224, 0.225]);
        assert!(Normalization::new([0.0; 3], [0.0; 3]).is_err());
    }

    #[test]
    fn normalize_pixel_maps_0_255_to_unit() {
        assert_eq!(normalize_pixel(255, 0.5, 0.5), 1.0);
        assert_eq!(normalize_pixel(0, 0.5, 0.5), -1.0);
        assert_eq!(normalize_pixel(128, 0.0, 1.0), 128.0 / 255.0);
    }

    #[test]
    fn preprocess_builds_chw_tensor_with_normalized_values() {
        // 2x2 uniform red upscaled to a 4x4 canvas (scale 2, no padding).
        let src: Vec<u8> = (0..4).flat_map(|_| [255u8, 0, 0]).collect();
        let mut out = vec![0f32; 3 * 16];
        let lb = preprocess_rgb8(
            &src,
            res(2, 2),
            res(4, 4),
            Normalization::default(),
            &mut out,
        )
        .unwrap();
        assert_eq!(lb.inner, res(4, 4));
        assert_eq!(out.len(), 48);

        let r = normalize_pixel(255, 0.485, 0.229);
        let g = normalize_pixel(0, 0.456, 0.224);
        let b = normalize_pixel(0, 0.406, 0.225);
        // CHW: R plane first, then G, then B — each 4x4.
        assert!(out[..16].iter().all(|&v| (v - r).abs() < 1e-5));
        assert!(out[16..32].iter().all(|&v| (v - g).abs() < 1e-5));
        assert!(out[32..48].iter().all(|&v| (v - b).abs() < 1e-5));
    }

    #[test]
    fn preprocess_zero_fills_padding() {
        // 2x1 into 4x4: scale 2, inner 4x2, pad_y 1.
        let src = vec![10u8, 20, 30, 200, 210, 220];
        let mut out = vec![1f32; 3 * 16];
        preprocess_rgb8(
            &src,
            res(2, 1),
            res(4, 4),
            Normalization::default(),
            &mut out,
        )
        .unwrap();
        for plane in 0..3 {
            let base = plane * 16;
            // Padded rows 0 and 3 are zero; inner rows 1-2 carry data.
            assert_eq!(out[base..base + 4], [0.0; 4], "top pad row");
            assert_eq!(out[base + 12..base + 16], [0.0; 4], "bottom pad row");
            assert!(out[base + 4..base + 12].iter().any(|&v| v != 0.0));
        }
    }

    #[test]
    fn bilinear_resize_identity_preserves_pixels() {
        let src: Vec<u8> = (0..12).collect();
        let mut out = vec![0f32; 12];
        preprocess_rgb8(
            &src,
            res(2, 2),
            res(2, 2),
            Normalization {
                mean: [0.0; 3],
                std: [1.0; 3],
            },
            &mut out,
        )
        .unwrap();
        // Identity resize: pixel (x,y) maps to itself.
        for y in 0..2 {
            for x in 0..2 {
                let i = (y * 2 + x) * 3;
                assert_eq!(out[y * 2 + x], src[i] as f32 / 255.0);
                assert_eq!(out[4 + y * 2 + x], src[i + 1] as f32 / 255.0);
                assert_eq!(out[8 + y * 2 + x], src[i + 2] as f32 / 255.0);
            }
        }
    }
}
