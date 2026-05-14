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
//! Byte order: the detector defaults to BGR (matching
//! [`crate::keyframe::preprocess`]); pass
//! [`ChannelOrder::Rgb`](crate::frame::ChannelOrder) via
//! [`Options::with_channel_order`] for native-RGB inputs. The `rg`
//! moment is symmetric under R/B swap but `yb` is not — picking the
//! wrong order produces a wrong colourfulness score, not a panic.
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

use crate::frame::{ChannelOrder, RgbFrame};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Options for the colorfulness detector.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Options {
  use_simd: bool,
  channel_order: ChannelOrder,
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
    Self {
      use_simd: true,
      channel_order: ChannelOrder::Bgr,
    }
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

  /// Sets the byte order of the packed-pixel input that
  /// [`Detector::observe_rgb`] will receive. Defaults to
  /// [`ChannelOrder::Bgr`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_channel_order(mut self, order: ChannelOrder) -> Self {
    self.channel_order = order;
    self
  }

  /// Returns the byte order the detector will read from each pixel.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn channel_order(&self) -> ChannelOrder {
    self.channel_order
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
  /// score; a uniform gray frame yields ~0. Reads channel positions
  /// according to [`Options::channel_order`].
  pub fn observe_rgb(&mut self, rgb: RgbFrame<'_>) -> f32 {
    crate::arch::colorfulness(
      rgb.data(),
      rgb.width() as usize,
      rgb.height() as usize,
      rgb.stride() as usize,
      self.opts.channel_order,
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

  #[test]
  fn options_default_is_bgr() {
    assert_eq!(Options::default().channel_order(), ChannelOrder::Bgr);
  }

  #[test]
  fn options_with_channel_order_round_trips() {
    let o = Options::new().with_channel_order(ChannelOrder::Rgb);
    assert_eq!(o.channel_order(), ChannelOrder::Rgb);
  }

  #[test]
  fn rgb_detector_matches_swapped_bgr_detector() {
    // Construct a non-trivial coloured frame, then run both
    // detectors on byte-swapped variants — outputs must agree.
    let (w, ht) = (32u32, 16u32);
    let mut bgr = vec![0u8; (w * ht * 3) as usize];
    let mut rng = 0xC0FFEE_u32;
    for v in bgr.iter_mut() {
      rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
      *v = (rng >> 24) as u8;
    }
    let mut rgb = bgr.clone();
    for chunk in rgb.chunks_exact_mut(3) {
      chunk.swap(0, 2);
    }

    let bgr_frame = RgbFrame::new(&bgr, w, ht, w * 3, timestamp());
    let rgb_frame = RgbFrame::new(&rgb, w, ht, w * 3, timestamp());

    let mut det_bgr = Detector::new(Options::default().with_simd(false));
    let mut det_rgb = Detector::new(
      Options::default()
        .with_simd(false)
        .with_channel_order(ChannelOrder::Rgb),
    );
    let c_bgr = det_bgr.observe_rgb(bgr_frame);
    let c_rgb = det_rgb.observe_rgb(rgb_frame);
    assert!(
      (c_bgr - c_rgb).abs() < 1e-3,
      "channel-order parity broken: bgr={c_bgr} rgb={c_rgb}"
    );
  }
}
