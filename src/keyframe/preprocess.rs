//! Shared per-frame preprocessing utilities for the keyframe detectors.
//!
//! Each utility owns its scratch buffers and exposes a `run(&mut self, …)`
//! method that produces a borrowed frame view. Buffers grow monotonically
//! — no per-frame allocation after the first sufficiently large call.
//!
//! # Typical use
//!
//! ```no_run
//! use core::num::{NonZeroI32, NonZeroU32};
//! use scenesdetect::frame::{RgbFrame, Timebase, Timestamp};
//! use scenesdetect::keyframe::preprocess::{Downscaler, LumaConverter, HsvConverter};
//!
//! let mut dn = Downscaler::new(NonZeroU32::new(256).unwrap());
//! let mut lc = LumaConverter::new();
//! let mut hc = HsvConverter::new();
//!
//! # let bytes: Vec<u8> = vec![0; 1920 * 1080 * 3];
//! # let tb = Timebase::new(1, NonZeroI32::new(1_000_000).unwrap());
//! # let frame = RgbFrame::new(&bytes, 1920, 1080, 1920 * 3, Timestamp::new(0, tb));
//! let small_bgr = dn.run(frame);        // 256-px longest side
//! let luma      = lc.run(small_bgr);    // BT.601 weighted sum
//! let hsv       = hc.run(small_bgr);    // OpenCV 8-bit HSV planes
//! ```
//!
//! All three inputs and outputs carry through the frame's
//! [`Timestamp`](crate::frame::Timestamp) unchanged — each utility is
//! a pure transformation on the pixels.
//!
//! # Byte order
//!
//! [`LumaConverter`] and [`HsvConverter`] each carry an explicit
//! [`ChannelOrder`](crate::frame::ChannelOrder) selecting `BGR` (the
//! default, matching OpenCV / the `content` detector) or `RGB`.
//! [`Downscaler`] is byte-order-agnostic — it operates per-channel and
//! preserves whichever order was passed in.

use core::num::NonZeroU32;

use fast_image_resize::{
  FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer,
  images::{Image, ImageRef},
};

use std::vec::Vec;

use crate::frame::{ChannelOrder, HsvFrame, LumaFrame, RgbFrame};

// ---- Downscaler -------------------------------------------------------------

/// Scales a packed 24-bit BGR frame so its longest side equals `target_dim`,
/// preserving aspect ratio. Uses [`ResizeAlg::Convolution`] with a Box
/// filter — the closest analogue to OpenCV's `INTER_AREA` and the right
/// choice for downscaling source material for perceptual quality metrics.
///
/// When the input's longest side is already `<= target_dim`, `run`
/// short-circuits and returns the input frame unchanged — no copy, no
/// resize work.
///
/// Internally owns two scratch buffers: one for re-packing stride-padded
/// source rows (only used when `src.stride() != 3 * src.width()`) and one
/// for the resized output. Both grow to the largest size seen.
///
/// # SIMD
///
/// There is no `use_simd` flag on [`Downscaler`] — unlike
/// [`LumaConverter`] and [`HsvConverter`] — because
/// [`fast_image_resize`] handles SIMD dispatch internally and does not
/// expose a disable switch. In testing contexts where fully scalar
/// output is required (e.g. bit-exact comparisons across targets), do
/// the downscale yourself and feed the converters directly.
pub struct Downscaler {
  target_dim: NonZeroU32,
  options: ResizeOptions,
  resizer: Resizer,
  src_tight: Vec<u8>,
  dst: Vec<u8>,
}

impl Downscaler {
  /// Creates a new [`Downscaler`] that targets the given longest-side
  /// dimension in pixels.
  pub fn new(target_dim: NonZeroU32) -> Self {
    Self {
      target_dim,
      options: ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Box)),
      resizer: Resizer::new(),
      src_tight: Vec::new(),
      dst: Vec::new(),
    }
  }

  /// Returns the configured longest-side target dimension.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn target_dim(&self) -> NonZeroU32 {
    self.target_dim
  }

  /// Downscales `src` so its longest side is [`target_dim`](Self::target_dim).
  ///
  /// Short-circuits (returns `src` as-is) when it is already small enough.
  /// Otherwise the result borrows from an internal scratch buffer — the
  /// caller may not hold two scratch-backed results from the same
  /// `Downscaler` at once.
  pub fn run<'a>(&'a mut self, src: RgbFrame<'a>) -> RgbFrame<'a> {
    let w = src.width();
    let h = src.height();
    let target = self.target_dim.get();

    // Preserve degenerate empty frames as-is. `RgbFrame` allows
    // `width == 0` or `height == 0`; without this short-circuit
    // `compute_resize_dims` would clamp the zero axis up to 1 and the
    // resizer would fabricate up to `target` pixels of synthetic
    // content from an empty source.
    if w == 0 || h == 0 {
      return src;
    }

    // Input already small enough — nothing to do.
    if w.max(h) <= target {
      return src;
    }

    let (new_w, new_h) = compute_resize_dims(w, h, target);
    let dst_size = (new_w as usize)
      .checked_mul(new_h as usize)
      .and_then(|v| v.checked_mul(3))
      .expect("output frame size fits in usize");
    let row_bytes = (w as usize) * 3;
    let src_tight_size = row_bytes * (h as usize);
    let needs_repack = src.stride() as usize != row_bytes;

    // Grow scratch buffers ahead of the borrow-sensitive resize call.
    if needs_repack && self.src_tight.len() < src_tight_size {
      self.src_tight.resize(src_tight_size, 0);
    }
    if self.dst.len() < dst_size {
      self.dst.resize(dst_size, 0);
    }

    // fast_image_resize demands tight-packed input. If the source row
    // stride matches width*3 we can borrow directly; otherwise rebundle
    // into `self.src_tight` first.
    let src_bytes: &[u8] = if !needs_repack {
      &src.data()[..src_tight_size]
    } else {
      let src_data = src.data();
      let src_stride = src.stride() as usize;
      for y in 0..h as usize {
        let soff = y * src_stride;
        let doff = y * row_bytes;
        self.src_tight[doff..doff + row_bytes].copy_from_slice(&src_data[soff..soff + row_bytes]);
      }
      &self.src_tight[..src_tight_size]
    };

    let src_img =
      ImageRef::new(w, h, src_bytes, PixelType::U8x3).expect("valid tight-packed BGR source");
    {
      let mut dst_img =
        Image::from_slice_u8(new_w, new_h, &mut self.dst[..dst_size], PixelType::U8x3)
          .expect("valid BGR destination slice");
      self
        .resizer
        .resize(&src_img, &mut dst_img, Some(&self.options))
        .expect("resize cannot fail for validated U8x3 inputs");
    }

    RgbFrame::new(
      &self.dst[..dst_size],
      new_w,
      new_h,
      new_w * 3,
      src.timestamp(),
    )
  }
}

/// Returns `(new_w, new_h)` such that `max(new_w, new_h) == target` and the
/// aspect ratio of `(w, h)` is preserved. Both outputs are clamped to at
/// least 1 so zero-dimension degeneracies are avoided.
fn compute_resize_dims(w: u32, h: u32, target: u32) -> (u32, u32) {
  debug_assert!(
    w.max(h) > target,
    "caller must short-circuit when input is already small"
  );
  if w >= h {
    let new_h = ((h as u64 * target as u64) / w as u64).max(1) as u32;
    (target, new_h)
  } else {
    let new_w = ((w as u64 * target as u64) / h as u64).max(1) as u32;
    (new_w, target)
  }
}

// ---- LumaConverter ----------------------------------------------------------

/// Converts a packed 24-bit BGR / RGB frame to a single-plane 8-bit BT.601 luma
/// buffer ([`crate::frame::convert::bgr_to_luma_with_order`]), reading channel
/// positions according to the configured [`ChannelOrder`].
///
/// Owns a single `Vec<u8>` scratch that grows to the largest frame it has
/// seen. The returned [`LumaFrame`] borrows from this scratch.
pub struct LumaConverter {
  scratch: Vec<u8>,
  use_simd: bool,
  channel_order: ChannelOrder,
}

impl Default for LumaConverter {
  fn default() -> Self {
    Self::new()
  }
}

impl LumaConverter {
  /// Creates a new converter with SIMD enabled and BGR byte order.
  pub const fn new() -> Self {
    Self {
      scratch: Vec::new(),
      use_simd: true,
      channel_order: ChannelOrder::Bgr,
    }
  }

  /// Sets whether to dispatch to SIMD backends when available.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_simd(mut self, on: bool) -> Self {
    self.use_simd = on;
    self
  }

  /// Sets the byte order of the packed input. Defaults to
  /// [`ChannelOrder::Bgr`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_channel_order(mut self, order: ChannelOrder) -> Self {
    self.channel_order = order;
    self
  }

  /// Returns the byte order this converter will interpret.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn channel_order(&self) -> ChannelOrder {
    self.channel_order
  }

  /// Converts the packed-pixel input to luma and returns a
  /// [`LumaFrame`] borrowing from this converter's scratch. Reads
  /// channel positions according to [`Self::channel_order`] — `src`
  /// may carry either BGR or RGB byte order.
  pub fn run<'a>(&'a mut self, src: RgbFrame<'_>) -> LumaFrame<'a> {
    let w = src.width();
    let h = src.height();
    let size = (w as usize)
      .checked_mul(h as usize)
      .expect("luma plane size fits in usize");

    if self.scratch.len() < size {
      self.scratch.resize(size, 0);
    }

    crate::frame::convert::bgr_to_luma_with_order(
      &mut self.scratch[..size],
      src.data(),
      w,
      h,
      src.stride(),
      self.channel_order,
      self.use_simd,
    );

    LumaFrame::new(&self.scratch[..size], w, h, w, src.timestamp())
  }
}

// ---- HsvConverter -----------------------------------------------------------

/// Converts a packed 24-bit BGR / RGB frame to three planar 8-bit HSV buffers
/// ([`crate::frame::convert::bgr_to_hsv_planes_with_order`]), reading channel
/// positions according to the configured [`ChannelOrder`].
///
/// Owns three `Vec<u8>` scratches — one per plane — that grow to the
/// largest frame seen. The returned [`HsvFrame`] borrows from them.
pub struct HsvConverter {
  h: Vec<u8>,
  s: Vec<u8>,
  v: Vec<u8>,
  use_simd: bool,
  channel_order: ChannelOrder,
}

impl Default for HsvConverter {
  fn default() -> Self {
    Self::new()
  }
}

impl HsvConverter {
  /// Creates a new converter with SIMD enabled and BGR byte order.
  pub const fn new() -> Self {
    Self {
      h: Vec::new(),
      s: Vec::new(),
      v: Vec::new(),
      use_simd: true,
      channel_order: ChannelOrder::Bgr,
    }
  }

  /// Sets whether to dispatch to SIMD backends when available.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_simd(mut self, on: bool) -> Self {
    self.use_simd = on;
    self
  }

  /// Sets the byte order of the packed input. Defaults to
  /// [`ChannelOrder::Bgr`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_channel_order(mut self, order: ChannelOrder) -> Self {
    self.channel_order = order;
    self
  }

  /// Returns the byte order this converter will interpret.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn channel_order(&self) -> ChannelOrder {
    self.channel_order
  }

  /// Converts the packed-pixel input to planar HSV and returns an
  /// [`HsvFrame`] borrowing from this converter's scratch. Reads
  /// channel positions according to [`Self::channel_order`] — `src`
  /// may carry either BGR or RGB byte order.
  pub fn run<'a>(&'a mut self, src: RgbFrame<'_>) -> HsvFrame<'a> {
    let w = src.width();
    let h = src.height();
    let size = (w as usize)
      .checked_mul(h as usize)
      .expect("HSV plane size fits in usize");

    if self.h.len() < size {
      self.h.resize(size, 0);
    }
    if self.s.len() < size {
      self.s.resize(size, 0);
    }
    if self.v.len() < size {
      self.v.resize(size, 0);
    }

    crate::frame::convert::bgr_to_hsv_planes_with_order(
      &mut self.h[..size],
      &mut self.s[..size],
      &mut self.v[..size],
      src.data(),
      w,
      h,
      src.stride(),
      self.channel_order,
      self.use_simd,
    );

    HsvFrame::new(
      &self.h[..size],
      &self.s[..size],
      &self.v[..size],
      w,
      h,
      w,
      src.timestamp(),
    )
  }
}

// -----------------------------------------------------------------------------

// Miri can't model the NEON / AVX intrinsics that `fast_image_resize`
// uses internally on aarch64 / x86_64 hosts (e.g.
// `llvm.aarch64.neon.st1x4.v4i32.p0` raises "unsupported foreign
// function"). Every `Downscaler` test routes through that crate, so
// gate the whole test module out under Miri rather than scattering
// `cfg_attr(miri, ignore)` over each test. Regular `cargo test` runs
// the suite as usual.
#[cfg(all(test, feature = "std", not(miri)))]
mod tests {
  use super::*;
  use crate::frame::{Timebase, Timestamp};
  use core::num::NonZeroI32;
  use std::vec;

  fn nz(n: u32) -> NonZeroU32 {
    NonZeroU32::new(n).expect("non-zero")
  }

  // `Downscaler` counts pixels (`NonZeroU32`); a `Timebase` denominator is
  // signed (`NonZeroI32`) — two different non-zero types, one per axis.
  fn nzi(n: i32) -> NonZeroI32 {
    NonZeroI32::new(n).expect("non-zero")
  }

  fn timestamp() -> Timestamp {
    Timestamp::new(42, Timebase::new(1, nzi(1000)))
  }

  fn make_bgr(w: usize, h: usize, seed: u32) -> std::vec::Vec<u8> {
    let mut buf = vec![0u8; w * h * 3];
    let mut rng = seed;
    for v in buf.iter_mut() {
      rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
      *v = (rng >> 24) as u8;
    }
    buf
  }

  // ----- compute_resize_dims --------------------------------------------------

  #[test]
  fn resize_dims_preserves_aspect_landscape() {
    // 1920×1080 → longest side 256 → (256, 144).
    assert_eq!(compute_resize_dims(1920, 1080, 256), (256, 144));
  }

  #[test]
  fn resize_dims_preserves_aspect_portrait() {
    // 1080×1920 → longest side 256 → (144, 256).
    assert_eq!(compute_resize_dims(1080, 1920, 256), (144, 256));
  }

  #[test]
  fn resize_dims_square() {
    assert_eq!(compute_resize_dims(2048, 2048, 256), (256, 256));
  }

  #[test]
  fn resize_dims_clamped_to_one_for_extreme_aspect() {
    // Extreme portrait: longer side scales to target, shorter clamps to 1.
    let (w, h) = compute_resize_dims(1, 2048, 4);
    assert_eq!(h, 4);
    assert_eq!(w, 1);
  }

  // ----- Downscaler -----------------------------------------------------------

  #[test]
  fn downscaler_shortcircuits_when_already_small() {
    let (w, h) = (100usize, 80usize);
    let src = make_bgr(w, h, 0x1);
    let frame = RgbFrame::new(&src, w as u32, h as u32, (w * 3) as u32, timestamp());

    let mut dn = Downscaler::new(nz(256));
    let out = dn.run(frame);

    // Should be byte-identical: we passed the original through.
    assert_eq!(out.width(), frame.width());
    assert_eq!(out.height(), frame.height());
    assert_eq!(out.stride(), frame.stride());
    assert_eq!(out.data().as_ptr(), frame.data().as_ptr());
  }

  #[test]
  fn downscaler_reduces_longest_side_to_target() {
    let (w, h) = (640usize, 360usize);
    let src = make_bgr(w, h, 0x2);
    let frame = RgbFrame::new(&src, w as u32, h as u32, (w * 3) as u32, timestamp());

    let mut dn = Downscaler::new(nz(256));
    let out = dn.run(frame);

    assert_eq!(out.width(), 256);
    assert_eq!(out.height(), 144);
    assert_eq!(out.stride(), 256 * 3);
    assert_eq!(out.data().len(), 256 * 144 * 3);
    assert_eq!(out.timestamp(), timestamp());
  }

  #[test]
  fn downscaler_handles_stride_padded_source() {
    // 320×180 pixel content, 1024-byte row stride (padding).
    let (w, h) = (320usize, 180usize);
    let stride = 1024usize;
    let mut src = vec![0u8; stride * h];
    let fill = make_bgr(w, h, 0x3);
    for y in 0..h {
      src[y * stride..y * stride + w * 3].copy_from_slice(&fill[y * w * 3..(y + 1) * w * 3]);
    }
    let frame = RgbFrame::new(&src, w as u32, h as u32, stride as u32, timestamp());

    let mut dn = Downscaler::new(nz(256));
    let out = dn.run(frame);

    assert_eq!(out.width(), 256);
    assert_eq!(out.height(), 144);
    // Same content as an unpadded run.
    let mut dn2 = Downscaler::new(nz(256));
    let frame2 = RgbFrame::new(&fill, w as u32, h as u32, (w * 3) as u32, timestamp());
    let out2 = dn2.run(frame2);
    assert_eq!(out.data(), out2.data());
  }

  #[test]
  fn downscaler_reuses_scratch_across_calls() {
    let (w, h) = (640usize, 360usize);
    let src = make_bgr(w, h, 0x4);
    let frame = RgbFrame::new(&src, w as u32, h as u32, (w * 3) as u32, timestamp());

    let mut dn = Downscaler::new(nz(256));
    {
      let _ = dn.run(frame);
    }
    let ptr_after_first = dn.dst.as_ptr();
    let len_after_first = dn.dst.len();
    {
      let _ = dn.run(frame);
    }
    // No reallocation between runs of the same size.
    assert_eq!(dn.dst.as_ptr(), ptr_after_first);
    assert_eq!(dn.dst.len(), len_after_first);
  }

  #[test]
  fn downscaler_preserves_zero_width_frame() {
    // 0×400 input with target 256 must NOT fabricate a 1×256 output.
    // The earlier short-circuit had `w.max(h) > target` for this
    // shape (400 > 256), so without the explicit zero-axis guard
    // `compute_resize_dims` would clamp the zero axis up to 1.
    let src = vec![0u8; 0];
    let frame = RgbFrame::new(&src, 0, 400, 0, timestamp());
    let mut dn = Downscaler::new(nz(256));
    let out = dn.run(frame);
    assert_eq!(out.width(), 0, "zero-width input must stay zero-width");
    assert_eq!(out.height(), 400);
    assert_eq!(out.data().as_ptr(), frame.data().as_ptr());
  }

  #[test]
  fn downscaler_preserves_zero_height_frame() {
    let src = vec![0u8; 0];
    let frame = RgbFrame::new(&src, 400, 0, 400 * 3, timestamp());
    let mut dn = Downscaler::new(nz(256));
    let out = dn.run(frame);
    assert_eq!(out.height(), 0, "zero-height input must stay zero-height");
    assert_eq!(out.width(), 400);
    assert_eq!(out.data().as_ptr(), frame.data().as_ptr());
  }

  // ----- LumaConverter --------------------------------------------------------

  #[test]
  fn luma_converter_matches_frame_convert() {
    let (w, h) = (32usize, 16usize);
    let src = make_bgr(w, h, 0x5);
    let frame = RgbFrame::new(&src, w as u32, h as u32, (w * 3) as u32, timestamp());

    let mut lc = LumaConverter::new();
    let out = lc.run(frame);

    let mut reference = vec![0u8; w * h];
    crate::frame::convert::bgr_to_luma_with_order(
      &mut reference,
      &src,
      w as u32,
      h as u32,
      (w * 3) as u32,
      ChannelOrder::Bgr,
      false,
    );

    assert_eq!(out.data(), &reference[..]);
    assert_eq!(out.width(), w as u32);
    assert_eq!(out.height(), h as u32);
    assert_eq!(out.timestamp(), timestamp());
  }

  // ----- HsvConverter ---------------------------------------------------------

  #[test]
  fn hsv_converter_produces_three_planes() {
    let (w, h) = (32usize, 16usize);
    let src = make_bgr(w, h, 0x6);
    let frame = RgbFrame::new(&src, w as u32, h as u32, (w * 3) as u32, timestamp());

    let mut hc = HsvConverter::new();
    let out = hc.run(frame);

    assert_eq!(out.width(), w as u32);
    assert_eq!(out.height(), h as u32);
    assert_eq!(out.hue().len(), w * h);
    assert_eq!(out.saturation().len(), w * h);
    assert_eq!(out.value().len(), w * h);
    // Random BGR input should exercise all three planes.
    assert!(out.value().iter().any(|&v| v > 0));
  }

  #[test]
  fn hsv_converter_matches_frame_convert() {
    let (w, h) = (32usize, 16usize);
    let src = make_bgr(w, h, 0x7);
    let frame = RgbFrame::new(&src, w as u32, h as u32, (w * 3) as u32, timestamp());

    let mut hc = HsvConverter::new().with_simd(false);
    let out = hc.run(frame);

    let mut rh = vec![0u8; w * h];
    let mut rs = vec![0u8; w * h];
    let mut rv = vec![0u8; w * h];
    crate::frame::convert::bgr_to_hsv_planes_with_order(
      &mut rh,
      &mut rs,
      &mut rv,
      &src,
      w as u32,
      h as u32,
      (w * 3) as u32,
      ChannelOrder::Bgr,
      false,
    );

    assert_eq!(out.hue(), &rh[..]);
    assert_eq!(out.saturation(), &rs[..]);
    assert_eq!(out.value(), &rv[..]);
  }

  #[test]
  fn luma_converter_with_rgb_order_matches_swapped_bgr() {
    let (w, h) = (32usize, 16usize);
    let bgr = make_bgr(w, h, 0x123_4567);
    let mut rgb = bgr.clone();
    for chunk in rgb.chunks_exact_mut(3) {
      chunk.swap(0, 2);
    }
    let bgr_frame = RgbFrame::new(&bgr, w as u32, h as u32, (w * 3) as u32, timestamp());
    let rgb_frame = RgbFrame::new(&rgb, w as u32, h as u32, (w * 3) as u32, timestamp());

    let mut lc_bgr = LumaConverter::new().with_simd(false);
    let mut lc_rgb = LumaConverter::new()
      .with_simd(false)
      .with_channel_order(ChannelOrder::Rgb);
    let out_bgr = lc_bgr.run(bgr_frame).data().to_vec();
    let out_rgb = lc_rgb.run(rgb_frame).data().to_vec();
    assert_eq!(out_bgr, out_rgb);
  }

  #[test]
  fn hsv_converter_with_rgb_order_matches_swapped_bgr() {
    let (w, h) = (32usize, 16usize);
    let bgr = make_bgr(w, h, 0xFEED_FACE);
    let mut rgb = bgr.clone();
    for chunk in rgb.chunks_exact_mut(3) {
      chunk.swap(0, 2);
    }
    let bgr_frame = RgbFrame::new(&bgr, w as u32, h as u32, (w * 3) as u32, timestamp());
    let rgb_frame = RgbFrame::new(&rgb, w as u32, h as u32, (w * 3) as u32, timestamp());

    let mut hc_bgr = HsvConverter::new().with_simd(false);
    let mut hc_rgb = HsvConverter::new()
      .with_simd(false)
      .with_channel_order(ChannelOrder::Rgb);
    let (h_b, s_b, v_b) = {
      let out_bgr = hc_bgr.run(bgr_frame);
      (
        out_bgr.hue().to_vec(),
        out_bgr.saturation().to_vec(),
        out_bgr.value().to_vec(),
      )
    };
    let out_rgb = hc_rgb.run(rgb_frame);
    assert_eq!(out_rgb.hue(), &h_b[..]);
    assert_eq!(out_rgb.saturation(), &s_b[..]);
    assert_eq!(out_rgb.value(), &v_b[..]);
  }

  #[test]
  fn luma_converter_default_channel_order_is_bgr() {
    let lc = LumaConverter::new();
    assert_eq!(lc.channel_order(), ChannelOrder::Bgr);
  }

  #[test]
  fn hsv_converter_default_channel_order_is_bgr() {
    let hc = HsvConverter::new();
    assert_eq!(hc.channel_order(), ChannelOrder::Bgr);
  }
}
