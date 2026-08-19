//! Clipping-ratio detector.
//!
//! Reports the fraction of pixels in a packed 24-bit BGR (or RGB — the
//! `max(R, G, B)` reduction is byte-order-agnostic) frame whose brightest
//! channel is **clipped** — either crushed at the shadow end or blown at
//! the highlight end. The thresholds are fixed at compile time: a
//! channel max of `< 5` counts as crushed, `> 250` as blown.
//!
//! Unlike a simple mean-luma check, this catches two problem cases the
//! [luma mean](crate::keyframe::luma) cannot:
//!
//! - Saturated coloured highlights — e.g. a blown-out pure-red lamp that
//!   the BT.601 weighted luma still averages to a moderate value.
//! - Large patches of detail-less black.
//!
//! The [selection state machine](crate::keyframe::select) rejects frames
//! with a clipping ratio above a configurable fraction of the pixel count.
//!
//! # Example
//!
//! ```no_run
//! use core::num::NonZeroI32;
//! use scenesdetect::frame::{RgbFrame, Timebase, Timestamp};
//! use scenesdetect::keyframe::clipping::{Detector, Options};
//!
//! let mut det = Detector::new(Options::default());
//! # let bytes = vec![0u8; 256 * 144 * 3];
//! # let tb = Timebase::new(1, NonZeroI32::new(1_000_000).unwrap());
//! # let f = RgbFrame::new(&bytes, 256, 144, 256 * 3, Timestamp::new(0, tb));
//! let clipping = det.observe_rgb(f);
//! assert!((0.0..=1.0).contains(&clipping));
//! ```

use crate::frame::RgbFrame;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// Thresholds live as compile-time constants in the scalar kernel at
// `crate::arch::scalar::Scalar::clipping_count` (currently `< 5` for
// crushed shadows, `> 250` for blown highlights). They are fixed
// sensor-scale properties and not exposed as tunables.

/// Options for the clipping detector.
///
/// Currently only carries the `use_simd` flag for forward-compatibility;
/// the scalar path is always used in v0.1.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Options {
  use_simd: bool,
}

impl Default for Options {
  fn default() -> Self {
    Self { use_simd: true }
  }
}

impl Options {
  /// Creates a new [`Options`] matching [`Options::default`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self { use_simd: true }
  }

  /// Sets whether to dispatch to SIMD backends when available.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_simd(mut self, on: bool) -> Self {
    self.use_simd = on;
    self
  }

  /// Returns whether SIMD dispatch is currently enabled.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn use_simd(&self) -> bool {
    self.use_simd
  }
}

/// Pure-algo state machine that reduces an RGB/BGR frame to its
/// clipping ratio (a value in `[0.0, 1.0]`).
#[derive(Debug, Clone)]
pub struct Detector {
  opts: Options,
}

impl Detector {
  /// Creates a new detector with the supplied options.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(opts: Options) -> Self {
    Self { opts }
  }

  /// Returns the detector's current options.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn options(&self) -> &Options {
    &self.opts
  }

  /// Resets stream state. No-op today; reserved for future SIMD caches.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn clear(&mut self) {}

  /// Computes the clipping ratio of `rgb`. The ratio is in `[0.0, 1.0]`:
  /// `0.0` means no clipped pixels, `1.0` means every pixel is clipped
  /// at one of the two extremes.
  ///
  /// The caller already has the frame's timestamp via
  /// [`RgbFrame::timestamp`] — it is not re-emitted here.
  pub fn observe_rgb(&mut self, rgb: RgbFrame<'_>) -> f32 {
    let w = rgb.width() as usize;
    let h = rgb.height() as usize;
    let n = w.saturating_mul(h);
    if n == 0 {
      return 0.0;
    }
    let count = crate::arch::clipping_count(
      rgb.data(),
      rgb.width(),
      rgb.height(),
      rgb.stride(),
      self.opts.use_simd,
    );
    (count as f64 / n as f64) as f32
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::frame::{RgbFrame, Timebase, Timestamp};
  use core::num::NonZeroI32;
  use std::vec;

  fn nz(n: i32) -> NonZeroI32 {
    NonZeroI32::new(n).expect("non-zero")
  }

  fn timestamp() -> Timestamp {
    Timestamp::new(7, Timebase::new(1, nz(1000)))
  }

  #[test]
  fn options_default_enables_simd() {
    assert!(Options::default().use_simd());
  }

  #[test]
  fn options_builder_roundtrips() {
    let o = Options::new().with_simd(false);
    assert!(!o.use_simd());
  }

  #[test]
  fn all_black_pixels_are_fully_clipped() {
    let data = vec![0u8; 16 * 16 * 3];
    let f = RgbFrame::new(&data, 16, 16, 16 * 3, timestamp());
    let mut det = Detector::new(Options::default());
    let r = det.observe_rgb(f);
    assert_eq!(r, 1.0);
  }

  #[test]
  fn all_white_pixels_are_fully_clipped() {
    let data = vec![255u8; 16 * 16 * 3];
    let f = RgbFrame::new(&data, 16, 16, 16 * 3, timestamp());
    let mut det = Detector::new(Options::default());
    let r = det.observe_rgb(f);
    assert_eq!(r, 1.0);
  }

  #[test]
  fn mid_gray_is_not_clipped() {
    let data = vec![128u8; 16 * 16 * 3];
    let f = RgbFrame::new(&data, 16, 16, 16 * 3, timestamp());
    let mut det = Detector::new(Options::default());
    let r = det.observe_rgb(f);
    assert_eq!(r, 0.0);
  }

  #[test]
  fn boundary_values_behave_correctly() {
    // CLIP_LOW=5, CLIP_HIGH=250. A pixel is clipped iff max(R,G,B) is
    // strictly below 5 or strictly above 250. Construct one of each:
    //   pixel 0: max=4   → clipped
    //   pixel 1: max=5   → NOT clipped (boundary inclusive)
    //   pixel 2: max=250 → NOT clipped (boundary inclusive)
    //   pixel 3: max=251 → clipped
    // Expect ratio = 2/4 = 0.5.
    let w = 4u32;
    let h = 1u32;
    let mut data = vec![0u8; (w as usize) * (h as usize) * 3];
    data[0] = 4; // pixel 0 BGR = (4, 0, 0)
    data[3 + 1] = 5; // pixel 1 BGR = (0, 5, 0)
    data[2 * 3 + 2] = 250; // pixel 2 BGR = (0, 0, 250)
    data[3 * 3 + 2] = 251; // pixel 3 BGR = (0, 0, 251)

    let f = RgbFrame::new(&data, w, h, w * 3, timestamp());
    let mut det = Detector::new(Options::default());
    let r = det.observe_rgb(f);
    assert!((r - 0.5).abs() < 1e-6, "expected 0.5, got {r}");
  }

  #[test]
  fn saturated_red_is_clipped_despite_low_luma_mean() {
    // Pure-red pixels: R=255, G=B=0. BT.601 luma mean ≈ 76 (moderate),
    // but max(R,G,B)=255 > 250 → clipped.
    let w = 8u32;
    let h = 8u32;
    let mut data = vec![0u8; (w * h * 3) as usize];
    for i in 0..(w * h) as usize {
      data[i * 3 + 2] = 255; // R (BGR layout → byte 2 is R)
    }
    let f = RgbFrame::new(&data, w, h, w * 3, timestamp());
    let mut det = Detector::new(Options::default());
    let r = det.observe_rgb(f);
    assert_eq!(r, 1.0);
  }

  #[test]
  fn stride_padding_is_ignored() {
    // 4×2 pixels, stride=16 (4 padding bytes per row). Padding filled
    // with 255 (would otherwise register as clipped highlights).
    let w = 4usize;
    let h = 2usize;
    let stride = 16usize;
    let mut data = vec![255u8; stride * h];
    // Write mid-gray into pixel area only.
    for y in 0..h {
      for x in 0..w {
        data[y * stride + x * 3] = 100;
        data[y * stride + x * 3 + 1] = 100;
        data[y * stride + x * 3 + 2] = 100;
      }
    }
    let f = RgbFrame::new(&data, w as u32, h as u32, stride as u32, timestamp());
    let mut det = Detector::new(Options::default());
    let r = det.observe_rgb(f);
    assert_eq!(r, 0.0, "padding bytes leaked into clipping count");
  }

  #[test]
  fn byte_order_agnostic() {
    // BGR or RGB — max(R,G,B) is the same regardless of which of the
    // three positions holds which channel. Verify by swapping B and R
    // position and checking identical ratio.
    let w = 8u32;
    let h = 4u32;
    // First frame: BGR layout with saturated blue in channel 0.
    let mut data_bgr = vec![0u8; (w * h * 3) as usize];
    for i in 0..(w * h) as usize {
      data_bgr[i * 3] = 251;
    }
    // Second frame: "RGB" layout with same saturated-blue byte in
    // channel 2 instead. Ratio must match.
    let mut data_rgb = vec![0u8; (w * h * 3) as usize];
    for i in 0..(w * h) as usize {
      data_rgb[i * 3 + 2] = 251;
    }

    let mut det = Detector::new(Options::default());
    let r1 = det.observe_rgb(RgbFrame::new(&data_bgr, w, h, w * 3, timestamp()));
    let r2 = det.observe_rgb(RgbFrame::new(&data_rgb, w, h, w * 3, timestamp()));
    assert_eq!(r1, r2);
    assert_eq!(r1, 1.0);
  }
}
