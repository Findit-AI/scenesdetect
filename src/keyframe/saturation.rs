//! HSV saturation-plane variance detector.
//!
//! Reports the population variance of the saturation plane in OpenCV's
//! 8-bit HSV encoding (`S ∈ [0, 255]`). Pairs with the
//! [luma variance](crate::keyframe::luma) via the selection state
//! machine's AND-gate: a frame is flagged "flat" only when **both**
//! luma and saturation variance fall below their thresholds. This
//! prevents equiluminant multi-colour frames (same brightness, different
//! hues — e.g. a red/blue colour bar) from being rejected as featureless.
//!
//! Two entry points:
//!
//! - [`Detector::observe_hsv`]: caller already has HSV planes. No
//!   conversion work.
//! - [`Detector::observe_rgb`]: convenience that internally runs
//!   [`crate::frame::convert::bgr_to_hsv_planes`] and then takes the
//!   S-plane variance. Owns three scratch buffers (H, S, V) that grow
//!   monotonically.
//!
//! # Example
//!
//! ```no_run
//! use core::num::NonZeroU32;
//! use scenesdetect::frame::{HsvFrame, Timebase, Timestamp};
//! use scenesdetect::keyframe::saturation::{Detector, Options};
//!
//! let mut det = Detector::new(Options::default());
//!
//! # let h = vec![0u8; 256 * 144];
//! # let s = vec![200u8; 256 * 144];
//! # let v = vec![128u8; 256 * 144];
//! # let tb = Timebase::new(1, NonZeroU32::new(1_000_000).unwrap());
//! # let hsv = HsvFrame::new(&h, &s, &v, 256, 144, 256, Timestamp::new(0, tb));
//! let sat_var = det.observe_hsv(hsv);
//! assert!(sat_var >= 0.0);
//! ```

use crate::frame::{ChannelOrder, HsvFrame, RgbFrame};

use std::vec::Vec;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Options for the saturation detector.
///
/// Carries a `use_simd` flag and the input pixel byte order. Note the
/// flag's scope:
///
/// - On [`Detector::observe_rgb`], the flag **does** route through to
///   the internal BGR→HSV conversion, which dispatches to its hand-
///   written SIMD backends (see [`crate::frame::convert`]).
/// - The variance reduction itself (applied on both `observe_hsv` and
///   `observe_rgb`) is **always scalar** in v0.1 — when a SIMD
///   reduction kernel lands, the same flag will gate it.
///
/// So `use_simd = false` is the right setting for tests that want
/// deterministic scalar output; `use_simd = true` (default) is correct
/// for production.
///
/// `channel_order` selects whether [`Detector::observe_rgb`] interprets
/// each packed pixel as `B, G, R` (default) or `R, G, B`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Options {
  use_simd: bool,
  // `serde(default)` keeps pre-`channel_order` payloads
  // deserializing (they fall back to `ChannelOrder::Bgr`, matching the
  // pre-existing internal behaviour). Without this, upgrading the
  // crate would break any stored config.
  #[cfg_attr(feature = "serde", serde(default))]
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

  /// Sets the byte order of the packed input that
  /// [`Detector::observe_rgb`] will receive. Defaults to
  /// [`ChannelOrder::Bgr`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_channel_order(mut self, order: ChannelOrder) -> Self {
    self.channel_order = order;
    self
  }

  /// Returns the byte order the detector will read.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn channel_order(&self) -> ChannelOrder {
    self.channel_order
  }
}

/// Pure-algo state machine that reduces a frame to its S-plane variance.
///
/// The struct holds a trio of optional scratch buffers used only by
/// [`Detector::observe_rgb`]; [`Detector::observe_hsv`] allocates nothing.
#[derive(Debug, Clone)]
pub struct Detector {
  opts: Options,
  // Scratches used only by observe_rgb. Grow monotonically; the three
  // planes always share the same length.
  h_scratch: Vec<u8>,
  s_scratch: Vec<u8>,
  v_scratch: Vec<u8>,
}

impl Detector {
  /// Creates a new detector with the supplied options.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(opts: Options) -> Self {
    Self {
      opts,
      h_scratch: Vec::new(),
      s_scratch: Vec::new(),
      v_scratch: Vec::new(),
    }
  }

  /// Returns the detector's current options.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn options(&self) -> &Options {
    &self.opts
  }

  /// Resets stream state. No-op today; scratches are preserved across
  /// `clear` calls (growing them only on demand).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn clear(&mut self) {}

  /// Computes the population variance of `hsv`'s saturation plane.
  /// The caller already has the frame's timestamp via
  /// [`HsvFrame::timestamp`] — it is not re-emitted here.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn observe_hsv(&mut self, hsv: HsvFrame<'_>) -> f32 {
    plane_variance_scalar(
      hsv.saturation(),
      hsv.width() as usize,
      hsv.height() as usize,
      hsv.stride() as usize,
      self.opts.use_simd,
    )
  }

  /// Convenience: internally converts `rgb` to HSV via
  /// [`bgr_to_hsv_planes`](crate::frame::convert::bgr_to_hsv_planes)
  /// (honouring [`Options::channel_order`]), then returns the
  /// S-plane variance.
  pub fn observe_rgb(&mut self, rgb: RgbFrame<'_>) -> f32 {
    let w = rgb.width();
    let h = rgb.height();
    let size = (w as usize)
      .checked_mul(h as usize)
      .expect("plane size fits in usize");

    if self.h_scratch.len() < size {
      self.h_scratch.resize(size, 0);
    }
    if self.s_scratch.len() < size {
      self.s_scratch.resize(size, 0);
    }
    if self.v_scratch.len() < size {
      self.v_scratch.resize(size, 0);
    }

    crate::frame::convert::bgr_to_hsv_planes_with_order(
      &mut self.h_scratch[..size],
      &mut self.s_scratch[..size],
      &mut self.v_scratch[..size],
      rgb.data(),
      w,
      h,
      rgb.stride(),
      self.opts.channel_order,
      self.opts.use_simd,
    );

    // Internal scratches are tight (stride == width).
    plane_variance_scalar(
      &self.s_scratch[..size],
      w as usize,
      h as usize,
      w as usize,
      self.opts.use_simd,
    )
  }
}

/// Variance via the shared reduction; mean is computed as a by-product
/// and discarded here.
fn plane_variance_scalar(
  plane: &[u8],
  width: usize,
  height: usize,
  stride: usize,
  use_simd: bool,
) -> f32 {
  let (_mean, variance) =
    super::reduce::plane_mean_variance(plane, width, height, stride, use_simd);
  variance
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::frame::{HsvFrame, RgbFrame, Timebase, Timestamp};
  use core::num::NonZeroU32;
  use std::vec;

  fn nz(n: u32) -> NonZeroU32 {
    NonZeroU32::new(n).expect("non-zero")
  }

  fn timestamp() -> Timestamp {
    Timestamp::new(77, Timebase::new(1, nz(1000)))
  }

  fn tight_hsv<'a>(h: &'a [u8], s: &'a [u8], v: &'a [u8], w: u32, ht: u32) -> HsvFrame<'a> {
    HsvFrame::new(h, s, v, w, ht, w, timestamp())
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
  fn uniform_saturation_has_zero_variance() {
    let (w, ht) = (16u32, 8u32);
    let n = (w * ht) as usize;
    let h = vec![0u8; n];
    let s = vec![200u8; n];
    let v = vec![128u8; n];
    let mut det = Detector::new(Options::default());
    let var = det.observe_hsv(tight_hsv(&h, &s, &v, w, ht));
    assert!(var < 1e-3, "uniform S should have zero variance, got {var}");
  }

  #[test]
  fn bimodal_saturation_variance_matches_expected() {
    // Half pixels at S=0, half at S=200 → mean 100, var = 100² = 10000.
    let (w, ht) = (16u32, 16u32);
    let n = (w * ht) as usize;
    let h = vec![0u8; n];
    let mut s = vec![0u8; n];
    s.iter_mut().take(n).skip(n / 2).for_each(|v| *v = 200);
    let v = vec![128u8; n];

    let mut det = Detector::new(Options::default());
    let var = det.observe_hsv(tight_hsv(&h, &s, &v, w, ht));
    assert!((var - 10000.0).abs() < 1.0, "var off: {var}");
  }

  #[test]
  fn observe_hsv_honours_stride_padding() {
    // 4×2 pixels, stride 8. Padding bytes are S=255 (would inflate var
    // if leaked). Pixel S values are all 0 → expected var=0.
    let w = 4u32;
    let ht = 2u32;
    let stride = 8u32;
    let plane_len = (stride * ht) as usize;
    let mut s = vec![255u8; plane_len];
    for y in 0..ht as usize {
      for x in 0..w as usize {
        s[y * stride as usize + x] = 0;
      }
    }
    let h = vec![0u8; plane_len];
    let v = vec![0u8; plane_len];
    let hsv = HsvFrame::new(&h, &s, &v, w, ht, stride, timestamp());

    let mut det = Detector::new(Options::default());
    let var = det.observe_hsv(hsv);
    assert_eq!(var, 0.0, "padding bytes leaked into variance");
  }

  #[test]
  fn observe_rgb_matches_observe_hsv_on_equivalent_input() {
    // Fill a BGR frame with a varying pattern; compare observe_rgb
    // against observe_hsv on the same converted planes.
    let (w, ht) = (24u32, 16u32);
    let n = (w * ht) as usize;

    let mut bgr = vec![0u8; n * 3];
    for (i, chunk) in bgr.chunks_exact_mut(3).enumerate() {
      // Cycle through a few saturated colours to generate variance.
      match i % 4 {
        0 => {
          chunk[0] = 255;
          chunk[1] = 0;
          chunk[2] = 0;
        } // blue
        1 => {
          chunk[0] = 0;
          chunk[1] = 255;
          chunk[2] = 0;
        } // green
        2 => {
          chunk[0] = 0;
          chunk[1] = 0;
          chunk[2] = 255;
        } // red
        _ => {
          chunk[0] = 128;
          chunk[1] = 128;
          chunk[2] = 128;
        } // gray
      }
    }

    let frame = RgbFrame::new(&bgr, w, ht, w * 3, timestamp());

    let mut det_rgb = Detector::new(Options::default());
    let v_rgb = det_rgb.observe_rgb(frame);

    let mut h = vec![0u8; n];
    let mut s = vec![0u8; n];
    let mut v = vec![0u8; n];
    crate::frame::convert::bgr_to_hsv_planes_with_order(
      &mut h,
      &mut s,
      &mut v,
      &bgr,
      w,
      ht,
      w * 3,
      ChannelOrder::Bgr,
      false,
    );

    let mut det_hsv = Detector::new(Options::default().with_simd(false));
    let v_hsv = det_hsv.observe_hsv(tight_hsv(&h, &s, &v, w, ht));

    assert!(
      (v_rgb - v_hsv).abs() < 1e-2,
      "rgb path {v_rgb} != hsv path {v_hsv}"
    );
  }

  #[test]
  fn options_default_channel_order_is_bgr() {
    assert_eq!(Options::default().channel_order(), ChannelOrder::Bgr);
  }

  #[cfg(feature = "serde")]
  #[test]
  fn options_deserializes_legacy_payload_without_channel_order() {
    // Pre-`channel_order` JSON payloads only had `use_simd`. Must
    // continue to deserialize so existing stored configs survive a
    // crate upgrade; missing field falls back to `Bgr` via
    // `#[serde(default)]`.
    let legacy = r#"{"use_simd": true}"#;
    let opts: Options = serde_json::from_str(legacy).expect("legacy payload must deserialize");
    assert!(opts.use_simd());
    assert_eq!(opts.channel_order(), ChannelOrder::Bgr);
  }

  #[test]
  fn observe_rgb_with_rgb_order_matches_swapped_bgr() {
    // Build a varying BGR frame, mirror it byte-swapped as RGB,
    // run both through their respective detectors — the HSV path
    // produces the same S-plane (and thus the same variance).
    let (w, ht) = (24u32, 16u32);
    let n = (w * ht) as usize;
    let mut bgr = vec![0u8; n * 3];
    for (i, chunk) in bgr.chunks_exact_mut(3).enumerate() {
      match i % 4 {
        0 => {
          chunk[0] = 255;
          chunk[1] = 0;
          chunk[2] = 0;
        }
        1 => {
          chunk[0] = 0;
          chunk[1] = 255;
          chunk[2] = 0;
        }
        2 => {
          chunk[0] = 0;
          chunk[1] = 0;
          chunk[2] = 255;
        }
        _ => {
          chunk[0] = 128;
          chunk[1] = 128;
          chunk[2] = 128;
        }
      }
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
    let v_bgr = det_bgr.observe_rgb(bgr_frame);
    let v_rgb = det_rgb.observe_rgb(rgb_frame);
    assert!(
      (v_bgr - v_rgb).abs() < 1e-3,
      "channel-order parity broken: bgr={v_bgr} rgb={v_rgb}"
    );
  }

  #[test]
  fn observe_rgb_reuses_scratch_across_calls() {
    let (w, ht) = (64u32, 48u32);
    let n = (w * ht) as usize;
    let bgr = vec![42u8; n * 3];
    let frame = RgbFrame::new(&bgr, w, ht, w * 3, timestamp());

    let mut det = Detector::new(Options::default());
    let _ = det.observe_rgb(frame);
    let ptrs = (
      det.h_scratch.as_ptr(),
      det.s_scratch.as_ptr(),
      det.v_scratch.as_ptr(),
    );
    let lens = (
      det.h_scratch.len(),
      det.s_scratch.len(),
      det.v_scratch.len(),
    );
    let _ = det.observe_rgb(frame);
    assert_eq!(
      ptrs,
      (
        det.h_scratch.as_ptr(),
        det.s_scratch.as_ptr(),
        det.v_scratch.as_ptr()
      )
    );
    assert_eq!(
      lens,
      (
        det.h_scratch.len(),
        det.s_scratch.len(),
        det.v_scratch.len()
      )
    );
  }
}
