//! Hasler-Süßstrunk colorfulness detector.
//!
//! Reports a perceptually-grounded "how colorful is this frame"
//! score in approximately `[0, 200]` (higher = more colorful).
//! Operates on packed 24-bit BGR. Single pass over the pixels using
//! streaming Welford-style moments on `rg = R − G` and
//! `yb = ½(R + G) − B`:
//!
//! ```text
//!     σ_rgyb = √(σ²_rg + σ²_yb)
//!     μ_rgyb = √(μ²_rg + μ²_yb)
//!     C      = σ_rgyb + 0.3 · μ_rgyb
//! ```
//!
//! Byte order: the implementation treats the packed input as BGR
//! (matching [`crate::keyframe::preprocess`]). RGB callers must
//! swizzle before feeding data in (the `rg` channel is symmetric
//! under R/B swap but `yb` is not).
//!
//! # Example
//!
//! ```no_run
//! use core::num::NonZeroU32;
//! use scenesdetect::frame::{RgbFrame, Timebase, Timestamp};
//! use scenesdetect::keyframe::colorfulness::{Detector, Options};
//!
//! let mut det = Detector::new(Options::default());
//! # let bytes = vec![0u8; 256 * 144 * 3];
//! # let tb = Timebase::new(1, NonZeroU32::new(1_000_000).unwrap());
//! # let f = RgbFrame::new(&bytes, 256, 144, 256 * 3, Timestamp::new(0, tb));
//! let c = det.observe_rgb(f);
//! assert!(c >= 0.0);
//! ```

use crate::frame::RgbFrame;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Options for the colorfulness detector.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Options {
  use_simd: bool,
}

impl Default for Options {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn default() -> Self {
    Self::new()
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

/// Pure-algo state machine that reduces a BGR frame to its
/// Hasler-Süßstrunk colorfulness score.
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

  /// Computes the colourfulness of `rgb`. Returns a non-negative
  /// score; a uniform gray frame yields ~0.
  pub fn observe_rgb(&mut self, rgb: RgbFrame<'_>) -> f32 {
    crate::arch::colorfulness(
      rgb.data(),
      rgb.width() as usize,
      rgb.height() as usize,
      rgb.stride() as usize,
      self.opts.use_simd,
    )
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::frame::{RgbFrame, Timebase, Timestamp};
  use core::num::NonZeroU32;
  use std::vec;

  fn nz(n: u32) -> NonZeroU32 {
    NonZeroU32::new(n).expect("non-zero")
  }

  fn timestamp() -> Timestamp {
    Timestamp::new(0, Timebase::new(1, nz(1_000_000)))
  }

  #[test]
  fn options_default_enables_simd() {
    assert!(Options::default().use_simd());
  }

  #[test]
  fn uniform_gray_frame_has_zero_colorfulness() {
    let w = 16u32;
    let h = 16u32;
    let data = vec![128u8; (w * h * 3) as usize];
    let f = RgbFrame::new(&data, w, h, w * 3, timestamp());
    let mut det = Detector::new(Options::default());
    let c = det.observe_rgb(f);
    assert!(c.abs() < 1e-3);
  }

  #[test]
  fn pure_red_has_nonzero_colorfulness() {
    let w = 8u32;
    let h = 8u32;
    let mut data = vec![0u8; (w * h * 3) as usize];
    for i in 0..(w * h) as usize {
      data[i * 3 + 2] = 255;
    }
    let f = RgbFrame::new(&data, w, h, w * 3, timestamp());
    let mut det = Detector::new(Options::default());
    let c = det.observe_rgb(f);
    let expected = 0.3_f64 * (255.0_f64.powi(2) + 127.5_f64.powi(2)).sqrt();
    assert!(
      ((c as f64) - expected).abs() < 1e-2,
      "expected ~{expected}, got {c}"
    );
  }

  #[test]
  fn clear_is_noop() {
    let mut det = Detector::new(Options::default());
    det.clear();
    let data = vec![128u8; 16 * 16 * 3];
    let f = RgbFrame::new(&data, 16, 16, 16 * 3, timestamp());
    let c = det.observe_rgb(f);
    assert!(c.abs() < 1e-3);
  }
}
