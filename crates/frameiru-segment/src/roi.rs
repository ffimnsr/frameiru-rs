//! Dynamic Crop & Track (Region of Interest / ROI Zoom).
//!
//! Tracks the user's upper body / head bounding box across consecutive frames.
//! Instead of downscaling the entire wide camera room to 256x256,
//! it crops the active subject region with smooth padding and feeds that into
//! the segmenter.
//!
//! This maximizes the effective neural resolution on the face, hair, and torso
//! (up to 3x-4x more pixels on the subject) while cleanly zeroing out distant background.

use frameiru_core::buffer::Mask;
use frameiru_core::format::Resolution;
use frameiru_core::FrameBuffer;

/// Integer pixel rectangle defining an ROI sub-region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoiRect {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct RoiRectFloat {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

/// Dynamic ROI tracker with temporal smoothing and edge proximity detection.
#[derive(Debug, Clone)]
pub struct RoiTracker {
    current_roi: Option<RoiRectFloat>,
    min_size_ratio: f32,
    padding: f32,
    smooth_alpha: f32,
    missed_frames: usize,
    max_missed_frames: usize,
}

impl Default for RoiTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl RoiTracker {
    /// Creates a new ROI tracker with default portrait framing tuning.
    pub fn new() -> Self {
        Self {
            current_roi: None,
            min_size_ratio: 0.35,
            padding: 0.25,
            smooth_alpha: 0.20,
            missed_frames: 0,
            max_missed_frames: 5,
        }
    }

    /// Resets the tracker to full-frame mode.
    pub fn reset(&mut self) {
        self.current_roi = None;
        self.missed_frames = 0;
    }

    /// Returns the current active ROI integer rectangle if tracking is active.
    pub fn current_rect(&self, frame_res: Resolution) -> Option<RoiRect> {
        let r = self.current_roi?;
        let fw = frame_res.width as usize;
        let fh = frame_res.height as usize;
        if fw == 0 || fh == 0 {
            return None;
        }

        let x = (r.x.round() as usize).min(fw - 1);
        let y = (r.y.round() as usize).min(fh - 1);
        let width = (r.w.round() as usize).clamp(1, fw - x);
        let height = (r.h.round() as usize).clamp(1, fh - y);

        // If ROI is almost the entire frame, no need to crop
        if width >= (fw * 75 / 100) || height >= (fh * 75 / 100) {
            return None;
        }

        Some(RoiRect {
            x,
            y,
            width,
            height,
        })
    }

    /// Updates the tracker state from the latest detected mask.
    pub fn update_from_mask(&mut self, mask: &Mask) {
        let frame_res = mask.resolution;
        let fw = frame_res.width as f32;
        let fh = frame_res.height as f32;
        if fw < 1.0 || fh < 1.0 {
            return;
        }

        let bounds = Self::find_mask_bounds(mask, 0.25);
        let Some(bounds) = bounds else {
            self.missed_frames += 1;
            if self.missed_frames >= self.max_missed_frames {
                self.current_roi = None;
            }
            return;
        };

        self.missed_frames = 0;

        // If subject is close-up (fills >= 60% of frame), stay in full-frame mode
        // to prevent cutting off shoulders and torso.
        if (bounds.width as f32) >= fw * 0.60 || (bounds.height as f32) >= fh * 0.60 {
            self.current_roi = None;
            return;
        }

        // Calculate target crop with padding
        let pad_x = (bounds.width as f32) * self.padding;
        let pad_y = (bounds.height as f32) * self.padding;

        let min_w = fw * self.min_size_ratio;
        let min_h = fh * self.min_size_ratio;

        let raw_w = (bounds.width as f32 + 2.0 * pad_x).max(min_w);
        let raw_h = (bounds.height as f32 + 2.0 * pad_y).max(min_h);

        // Maintain square bounding box for 1:1 portrait neural networks
        let side = raw_w.max(raw_h).min(fw).min(fh);

        let center_x = bounds.x as f32 + (bounds.width as f32) * 0.5;
        let center_y = bounds.y as f32 + (bounds.height as f32) * 0.5;

        let target_x = (center_x - side * 0.5).clamp(0.0, (fw - side).max(0.0));
        let target_y = (center_y - side * 0.5).clamp(0.0, (fh - side).max(0.0));

        let target = RoiRectFloat {
            x: target_x,
            y: target_y,
            w: side,
            h: side,
        };

        // Smooth transition
        let smoothed = match self.current_roi {
            Some(prev) => {
                // If subject is near the crop edge, adapt faster to avoid truncating motion
                let margin = prev.w * 0.06;
                let near_edge = (bounds.x as f32) < (prev.x + margin)
                    || ((bounds.x + bounds.width) as f32) > (prev.x + prev.w - margin)
                    || (bounds.y as f32) < (prev.y + margin)
                    || ((bounds.y + bounds.height) as f32) > (prev.y + prev.h - margin);

                let alpha = if near_edge { 0.50 } else { self.smooth_alpha };
                RoiRectFloat {
                    x: prev.x + (target.x - prev.x) * alpha,
                    y: prev.y + (target.y - prev.y) * alpha,
                    w: prev.w + (target.w - prev.w) * alpha,
                    h: prev.h + (target.h - prev.h) * alpha,
                }
            }
            None => target,
        };

        self.current_roi = Some(smoothed);
    }

    /// Finds the bounding box of foreground pixels where alpha >= threshold.
    pub fn find_mask_bounds(mask: &Mask, threshold: f32) -> Option<RoiRect> {
        let w = mask.resolution.width as usize;
        let h = mask.resolution.height as usize;
        if w == 0 || h == 0 || mask.data.len() < w * h {
            return None;
        }

        let mut min_x = usize::MAX;
        let mut max_x = 0;
        let mut min_y = usize::MAX;
        let mut max_y = 0;
        let mut count = 0usize;

        // Step by 2 in x and y for speed (~4x fewer pixel reads, sub-pixel accurate enough)
        for y in (0..h).step_by(2) {
            let row = y * w;
            for x in (0..w).step_by(2) {
                if mask.data[row + x] >= threshold {
                    min_x = min_x.min(x);
                    max_x = max_x.max(x);
                    min_y = min_y.min(y);
                    max_y = max_y.max(y);
                    count += 1;
                }
            }
        }

        // Require at least 20 sampled positive points to filter camera sensor salt noise
        if count < 20 || min_x > max_x || min_y > max_y {
            None
        } else {
            Some(RoiRect {
                x: min_x,
                y: min_y,
                width: max_x - min_x + 1,
                height: max_y - min_y + 1,
            })
        }
    }
}

/// Extracts the RGB8 sub-region of `frame` defined by `roi`.
pub fn crop_frame(frame: &FrameBuffer, roi: RoiRect) -> FrameBuffer {
    let mut sub = FrameBuffer::new(frameiru_core::format::FrameMetadata {
        resolution: Resolution {
            width: roi.width as u32,
            height: roi.height as u32,
        },
        format: frame.metadata.format,
        sequence: frame.metadata.sequence,
        timestamp_us: frame.metadata.timestamp_us,
    });

    let src_w = frame.metadata.resolution.width as usize;
    let bpp = 3;
    let dst_stride = roi.width * bpp;

    for row in 0..roi.height {
        let src_off = ((roi.y + row) * src_w + roi.x) * bpp;
        let dst_off = row * dst_stride;
        sub.data[dst_off..dst_off + dst_stride]
            .copy_from_slice(&frame.data[src_off..src_off + dst_stride]);
    }

    sub
}

/// Pastes a cropped ROI mask back into a full-resolution mask canvas.
pub fn paste_mask_roi(full_mask: &mut Mask, roi_mask: &Mask, roi: RoiRect) {
    let full_w = full_mask.resolution.width as usize;
    let full_h = full_mask.resolution.height as usize;
    let area = full_w * full_h;
    if full_mask.data.len() < area {
        full_mask.data.resize(area, 0.0);
    }
    full_mask.data.fill(0.0);

    let rw = roi.width;
    let rh = roi.height;
    let feather = if rw > 16 && rh > 16 {
        12usize.min(rw / 8).min(rh / 8)
    } else {
        0
    };

    for row in 0..rh {
        if roi.y + row >= full_h {
            break;
        }
        let src_off = row * rw;
        let dst_off = (roi.y + row) * full_w + roi.x;
        let copy_len = rw.min(full_w.saturating_sub(roi.x));

        let dist_y = row.min(rh.saturating_sub(1 + row));
        let fy = if feather > 0 && dist_y < feather {
            dist_y as f32 / feather as f32
        } else {
            1.0
        };

        for col in 0..copy_len {
            let dist_x = col.min(copy_len.saturating_sub(1 + col));
            let fx = if feather > 0 && dist_x < feather {
                dist_x as f32 / feather as f32
            } else {
                1.0
            };
            let weight = fx * fy;
            full_mask.data[dst_off + col] = roi_mask.data[src_off + col] * weight;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use frameiru_core::format::{FrameMetadata, PixelFormat};

    #[test]
    fn find_bounds_detects_box_accurately() {
        let res = Resolution {
            width: 100,
            height: 100,
        };
        let mut mask = Mask::filled(res, 0.0);

        // Fill a region from (20, 30) to (50, 70)
        for y in 30..=70 {
            for x in 20..=50 {
                mask.data[y * 100 + x] = 1.0;
            }
        }

        let bounds = RoiTracker::find_mask_bounds(&mask, 0.5).expect("found bounds");
        assert!(bounds.x <= 20);
        assert!(bounds.y <= 30);
        assert!(bounds.x + bounds.width >= 50);
        assert!(bounds.y + bounds.height >= 70);
    }

    #[test]
    fn crop_and_paste_roundtrips_data() {
        let res = Resolution {
            width: 10,
            height: 10,
        };
        let mut frame = FrameBuffer::new(FrameMetadata {
            resolution: res,
            format: PixelFormat::Rgb8,
            sequence: 1,
            timestamp_us: 0,
        });
        // Fill frame with coordinates as colors
        for y in 0..10 {
            for x in 0..10 {
                let idx = (y * 10 + x) * 3;
                frame.data[idx] = x as u8;
                frame.data[idx + 1] = y as u8;
                frame.data[idx + 2] = 255;
            }
        }

        let roi = RoiRect {
            x: 2,
            y: 3,
            width: 4,
            height: 5,
        };
        let cropped = crop_frame(&frame, roi);
        assert_eq!(cropped.metadata.resolution.width, 4);
        assert_eq!(cropped.metadata.resolution.height, 5);

        // Verify top-left of cropped matches (2, 3)
        assert_eq!(cropped.data[0], 2);
        assert_eq!(cropped.data[1], 3);

        // Simulate mask for cropped
        let roi_mask = Mask::filled(cropped.metadata.resolution, 0.85);
        let mut full_mask = Mask::filled(res, 0.0);
        paste_mask_roi(&mut full_mask, &roi_mask, roi);

        // Check inside ROI
        for y in 3..8 {
            for x in 2..6 {
                assert_eq!(full_mask.data[y * 10 + x], 0.85);
            }
        }
        // Check outside ROI
        assert_eq!(full_mask.data[0], 0.0);
        assert_eq!(full_mask.data[9 * 10 + 9], 0.0);
    }
}
