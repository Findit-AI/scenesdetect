//! Frame-input types for the scene detectors.
//!
//! The time primitives ([`Timebase`](crate::frame::Timebase),
//! [`Timestamp`](crate::frame::Timestamp), and
//! [`TimeRange`](crate::frame::TimeRange)) live in the [`mediatime`] crate
//! and are re-exported here so existing imports (`crate::frame::Timestamp`
//! etc.) keep working. This module owns the frame-buffer types
//! ([`LumaFrame`](crate::frame::LumaFrame),
//! [`RgbFrame`](crate::frame::RgbFrame),
//! [`HsvFrame`](crate::frame::HsvFrame)) and their validation errors.

use derive_more::{Display, IsVariant};
use thiserror::Error;

pub use mediatime::{TimeRange, Timebase, Timestamp};

/// Color-space conversions for the frame types in this module.
///
/// Backed by the crate's SIMD kernels (NEON / SSSE3 / AVX2 / wasm-simd128)
/// with a scalar fallback, runtime-dispatched on x86 when `std` is enabled.
#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "std", feature = "alloc"))))]
pub mod convert;

// /// Adapters for the RGB-family frame types from the [`mediaframe`] crate.
// #[cfg(feature = "mediaframe")]
// #[cfg_attr(docsrs, doc(cfg(feature = "mediaframe")))]
// pub mod mediaframe;

/// The byte order of a packed 3-byte-per-pixel color frame.
///
/// [`RgbFrame`] itself is byte-order-agnostic — it only knows that each
/// pixel is 3 contiguous bytes. Detectors that interpret individual
/// channels (`Colorfulness`, `Saturation`, the HSV/luma converters)
/// need to know which byte position holds which color channel.
///
/// The default is [`ChannelOrder::Bgr`] — that was the original crate
/// convention. Callers feeding native-RGB data should explicitly pass
/// [`ChannelOrder::Rgb`] (via the detector's `Options::channel_order`
/// or the explicit `*_with_order` entry points) so the kernels read
/// the channels from the right positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ChannelOrder {
  /// First byte = blue, second = green, third = red.
  #[default]
  Bgr,
  /// First byte = red, second = green, third = blue.
  Rgb,
}

/// A frame containing YUV luma (Y-plane) data, along with its dimensions and
/// presentation timestamp.
///
/// `data` points to tightly packed 8-bit luma samples. Rows may be padded:
/// row `y` starts at byte offset `y * stride`, and only the first `width` bytes
/// of each row carry pixels. `stride` is always `>= width`.
#[derive(Debug, Clone, Copy)]
pub struct LumaFrame<'a> {
  data: &'a [u8],
  width: u32,
  height: u32,
  stride: u32,
  timestamp: Timestamp,
}

impl<'a> LumaFrame<'a> {
  /// Creates a new `LumaFrame`, validating dimensions.
  ///
  /// # Panics
  ///
  /// Panics if the frame is invalid. Prefer [`Self::try_new`] for runtime-validated
  /// inputs; this constructor is meant for call sites where validity is statically
  /// known (tests, fixtures, callers that already checked).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(
    data: &'a [u8],
    width: u32,
    height: u32,
    stride: u32,
    timestamp: Timestamp,
  ) -> Self {
    match Self::try_new(data, width, height, stride, timestamp) {
      Ok(f) => f,
      Err(_) => panic!("invalid LumaFrame dimensions or data length"),
    }
  }

  /// Creates a new `LumaFrame`, returning an error if dimensions are inconsistent.
  ///
  /// Validates:
  /// - `stride >= width` (padding is allowed; underflow is not)
  /// - `stride * height` fits in `usize`
  /// - `data.len() >= stride * height`
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    data: &'a [u8],
    width: u32,
    height: u32,
    stride: u32,
    timestamp: Timestamp,
  ) -> Result<Self, LumaFrameError> {
    if stride < width {
      return Err(LumaFrameError::StrideTooSmall { width, stride });
    }
    let expected = match (stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => return Err(LumaFrameError::DimensionsOverflow { stride, height }),
    };
    if data.len() < expected {
      return Err(LumaFrameError::DataTooShort {
        expected,
        actual: data.len(),
      });
    }
    Ok(Self {
      data,
      width,
      height,
      stride,
      timestamp,
    })
  }

  /// Returns the Y-plane bytes. Row `y` starts at byte offset `y * stride`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn data(&self) -> &'a [u8] {
    self.data
  }

  /// Returns the width of the frame in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn width(&self) -> u32 {
    self.width
  }

  /// Returns the height of the frame in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn height(&self) -> u32 {
    self.height
  }

  /// Returns the stride of the frame in bytes per row. May exceed `width` due
  /// to alignment padding.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn stride(&self) -> u32 {
    self.stride
  }

  /// Returns the presentation timestamp of the frame.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn timestamp(&self) -> Timestamp {
    self.timestamp
  }
}

/// A frame containing packed 24-bit RGB (or BGR) data, three interleaved
/// bytes per pixel, along with its dimensions and presentation timestamp.
///
/// This type is byte-order-agnostic: detectors that only care about overall
/// brightness (like [`threshold::Detector`](crate::threshold::Detector)) treat RGB and BGR
/// equivalently. For detectors that care about channel meaning (future
/// color-based detectors), the caller is responsible for ensuring the bytes
/// are in the expected order.
///
/// Rows may be padded: row `y` starts at byte offset `y * stride`, and only
/// the first `width * 3` bytes of each row carry pixel data. `stride` is
/// always `>= width * 3`.
#[derive(Debug, Clone, Copy)]
pub struct RgbFrame<'a> {
  data: &'a [u8],
  width: u32,
  height: u32,
  stride: u32,
  timestamp: Timestamp,
}

impl<'a> RgbFrame<'a> {
  /// Bytes per pixel for the packed RGB / BGR layout.
  pub const BYTES_PER_PIXEL: u32 = 3;

  /// Creates a new `RgbFrame`, validating dimensions.
  ///
  /// Prefer [`Self::try_new`] at runtime call sites where invalid data is
  /// possible; this constructor is meant for call sites where validity is
  /// statically known.
  ///
  /// # Panics
  ///
  /// Panics if the frame is invalid. See [`RgbFrameError`] for conditions.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(
    data: &'a [u8],
    width: u32,
    height: u32,
    stride: u32,
    timestamp: Timestamp,
  ) -> Self {
    match Self::try_new(data, width, height, stride, timestamp) {
      Ok(f) => f,
      Err(_) => panic!("invalid RgbFrame dimensions or data length"),
    }
  }

  /// Creates a new `RgbFrame`, returning an error if dimensions are inconsistent.
  ///
  /// Validates:
  /// - `stride >= width * 3` (padding is allowed; underflow is not)
  /// - `stride * height` fits in `usize`
  /// - `data.len() >= stride * height`
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    data: &'a [u8],
    width: u32,
    height: u32,
    stride: u32,
    timestamp: Timestamp,
  ) -> Result<Self, RgbFrameError> {
    let min_stride = match width.checked_mul(Self::BYTES_PER_PIXEL) {
      Some(v) => v,
      None => return Err(RgbFrameError::WidthOverflow { width }),
    };
    if stride < min_stride {
      return Err(RgbFrameError::StrideTooSmall {
        width,
        stride,
        min_stride,
      });
    }
    let expected = match (stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => return Err(RgbFrameError::DimensionsOverflow { stride, height }),
    };
    if data.len() < expected {
      return Err(RgbFrameError::DataTooShort {
        expected,
        actual: data.len(),
      });
    }
    Ok(Self {
      data,
      width,
      height,
      stride,
      timestamp,
    })
  }

  /// Returns the packed RGB bytes. Row `y` starts at byte offset `y * stride`;
  /// within each row, pixel `x` occupies bytes `x*3 .. x*3 + 3`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn data(&self) -> &'a [u8] {
    self.data
  }

  /// Returns the width of the frame in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn width(&self) -> u32 {
    self.width
  }

  /// Returns the height of the frame in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn height(&self) -> u32 {
    self.height
  }

  /// Returns the stride of the frame in bytes per row. May exceed
  /// `width * 3` due to alignment padding.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn stride(&self) -> u32 {
    self.stride
  }

  /// Returns the presentation timestamp of the frame.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn timestamp(&self) -> Timestamp {
    self.timestamp
  }
}

/// Error returned by [`RgbFrame::try_new`] when the provided dimensions or
/// data length are inconsistent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum RgbFrameError {
  /// `stride` was smaller than `width * 3`. Stride is the number of bytes
  /// per row including any padding, and must cover the pixel row (3 bytes
  /// per pixel).
  #[error("stride ({stride}) is smaller than width*3 ({min_stride})")]
  StrideTooSmall {
    /// The frame width in pixels.
    width: u32,
    /// The provided stride in bytes.
    stride: u32,
    /// The minimum acceptable stride (`width * 3`).
    min_stride: u32,
  },
  /// The provided byte slice was too short to hold `stride * height` bytes.
  #[error("data length {actual} is less than required {expected} bytes")]
  DataTooShort {
    /// Minimum required byte length.
    expected: usize,
    /// Actual byte length of `data`.
    actual: usize,
  },
  /// `width * BYTES_PER_PIXEL` (i.e. `width * 3`) overflowed `u32`.
  #[error("width ({width}) * 3 overflows u32")]
  WidthOverflow {
    /// The frame width in pixels.
    width: u32,
  },
  /// `stride * height` overflowed `usize` (can only happen on 32-bit
  /// targets with very large frames).
  #[error("frame dimensions overflow usize: stride ({stride}) * height ({height})")]
  DimensionsOverflow {
    /// The stride in bytes.
    stride: u32,
    /// The frame height in pixels.
    height: u32,
  },
}

/// A frame in HSV color space, stored as three separate 8-bit planes.
///
/// Follows OpenCV's 8-bit HSV encoding: `H ∈ [0, 179]` (hue in degrees
/// divided by 2 so it fits in `u8`), `S ∈ [0, 255]`, `V ∈ [0, 255]`.
///
/// This is the planar form produced by
/// `cv2.split(cv2.cvtColor(..., COLOR_BGR2HSV))` in Python. If your
/// producer hands you interleaved HSV triples, split them into planes
/// first.
///
/// All three planes share the same dimensions and stride, and row `y`
/// starts at byte offset `y * stride` in each plane.
#[derive(Debug, Clone, Copy)]
pub struct HsvFrame<'a> {
  h: &'a [u8],
  s: &'a [u8],
  v: &'a [u8],
  width: u32,
  height: u32,
  stride: u32,
  timestamp: Timestamp,
}

impl<'a> HsvFrame<'a> {
  /// Creates a new `HsvFrame`, validating dimensions of all three planes.
  ///
  /// # Panics
  ///
  /// Panics if any plane is invalid. See [`HsvFrameError`] for conditions.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(
    h: &'a [u8],
    s: &'a [u8],
    v: &'a [u8],
    width: u32,
    height: u32,
    stride: u32,
    timestamp: Timestamp,
  ) -> Self {
    match Self::try_new(h, s, v, width, height, stride, timestamp) {
      Ok(f) => f,
      Err(_) => panic!("invalid HsvFrame dimensions or data length"),
    }
  }

  /// Creates a new `HsvFrame`, returning an error if the three planes are
  /// inconsistent in size or if any is too short for the given dimensions.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    h: &'a [u8],
    s: &'a [u8],
    v: &'a [u8],
    width: u32,
    height: u32,
    stride: u32,
    timestamp: Timestamp,
  ) -> Result<Self, HsvFrameError> {
    if stride < width {
      return Err(HsvFrameError::StrideTooSmall { width, stride });
    }
    let expected = match (stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => return Err(HsvFrameError::DimensionsOverflow { stride, height }),
    };
    if h.len() < expected {
      return Err(HsvFrameError::PlaneTooShort {
        plane: HsvPlane::Hue,
        expected,
        actual: h.len(),
      });
    }
    if s.len() < expected {
      return Err(HsvFrameError::PlaneTooShort {
        plane: HsvPlane::Saturation,
        expected,
        actual: s.len(),
      });
    }
    if v.len() < expected {
      return Err(HsvFrameError::PlaneTooShort {
        plane: HsvPlane::Value,
        expected,
        actual: v.len(),
      });
    }
    Ok(Self {
      h,
      s,
      v,
      width,
      height,
      stride,
      timestamp,
    })
  }

  /// Returns the hue (H) plane, `[0, 179]` per OpenCV's 8-bit encoding.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn hue(&self) -> &'a [u8] {
    self.h
  }

  /// Returns the saturation (S) plane, `[0, 255]`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn saturation(&self) -> &'a [u8] {
    self.s
  }

  /// Returns the value / brightness (V) plane, `[0, 255]`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn value(&self) -> &'a [u8] {
    self.v
  }

  /// Returns the frame width in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn width(&self) -> u32 {
    self.width
  }

  /// Returns the frame height in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn height(&self) -> u32 {
    self.height
  }

  /// Returns the per-plane stride in bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn stride(&self) -> u32 {
    self.stride
  }

  /// Returns the presentation timestamp.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn timestamp(&self) -> Timestamp {
    self.timestamp
  }
}

/// Which plane of an [`HsvFrame`] failed validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Display)]
#[display("{}", self.as_str())]
pub enum HsvPlane {
  /// Hue plane.
  Hue,
  /// Saturation plane.
  Saturation,
  /// Value (brightness) plane.
  Value,
}

impl HsvPlane {
  /// Returns a human-friendly name for the plane.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Hue => "hue",
      Self::Saturation => "saturation",
      Self::Value => "value",
    }
  }
}

/// Error returned by [`HsvFrame::try_new`] when the planes are inconsistent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum HsvFrameError {
  /// `stride` was smaller than `width`.
  #[error("stride ({stride}) is smaller than width ({width})")]
  StrideTooSmall {
    /// The frame width in pixels.
    width: u32,
    /// The provided stride in bytes.
    stride: u32,
  },
  /// One of the planes was too short.
  #[error("{plane} plane has length {actual} but at least {expected} are required")]
  PlaneTooShort {
    /// Which plane had insufficient data.
    plane: HsvPlane,
    /// Minimum required byte length per plane.
    expected: usize,
    /// Actual byte length.
    actual: usize,
  },
  /// `stride * height` overflowed `usize`.
  #[error("frame dimensions overflow usize: stride ({stride}) * height ({height})")]
  DimensionsOverflow {
    /// The stride in bytes.
    stride: u32,
    /// The frame height in pixels.
    height: u32,
  },
}

/// Error returned by [`LumaFrame::try_new`] when the provided dimensions or
/// data length are inconsistent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum LumaFrameError {
  /// `stride` was smaller than `width`. Stride is the number of bytes per row
  /// including any padding, and must cover the pixel width.
  #[error("stride ({stride}) is smaller than width ({width})")]
  StrideTooSmall {
    /// The frame width in pixels.
    width: u32,
    /// The provided stride in bytes.
    stride: u32,
  },
  /// The provided byte slice was too short to hold `stride * height` bytes.
  #[error("data length {actual} is less than required {expected} bytes")]
  DataTooShort {
    /// Minimum required byte length.
    expected: usize,
    /// Actual byte length of `data`.
    actual: usize,
  },
  /// `stride * height` overflowed `usize` (can only happen on 32-bit targets
  /// with very large frames).
  #[error("frame dimensions overflow usize: stride ({stride}) * height ({height})")]
  DimensionsOverflow {
    /// The stride in bytes.
    stride: u32,
    /// The frame height in pixels.
    height: u32,
  },
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use core::num::NonZeroI32;

  const fn nz(n: i32) -> NonZeroI32 {
    match NonZeroI32::new(n) {
      Some(v) => v,
      None => panic!("zero"),
    }
  }

  #[test]
  fn luma_frame_basic() {
    let buf = [0u8; 64 * 48];
    let tb = Timebase::new(1, nz(1000));
    let f = LumaFrame::new(&buf, 64, 48, 64, Timestamp::new(0, tb));
    assert_eq!(f.width(), 64);
    assert_eq!(f.height(), 48);
    assert_eq!(f.stride(), 64);
    assert_eq!(f.data().len(), 64 * 48);
  }

  #[test]
  fn luma_frame_with_padding() {
    let buf = [0u8; 80 * 48];
    let tb = Timebase::new(1, nz(1000));
    let f = LumaFrame::new(&buf, 64, 48, 80, Timestamp::new(0, tb));
    assert_eq!(f.width(), 64);
    assert_eq!(f.stride(), 80);
  }

  #[test]
  #[should_panic(expected = "invalid LumaFrame")]
  fn luma_frame_new_panics_on_stride_less_than_width() {
    let buf = [0u8; 64 * 48];
    let tb = Timebase::new(1, nz(1000));
    let _ = LumaFrame::new(&buf, 64, 48, 32, Timestamp::new(0, tb));
  }

  #[test]
  #[should_panic(expected = "invalid LumaFrame")]
  fn luma_frame_new_panics_on_short_data() {
    let buf = [0u8; 10];
    let tb = Timebase::new(1, nz(1000));
    let _ = LumaFrame::new(&buf, 64, 48, 64, Timestamp::new(0, tb));
  }

  #[test]
  fn try_new_success() {
    let buf = [0u8; 80 * 48];
    let tb = Timebase::new(1, nz(1000));
    let f = LumaFrame::try_new(&buf, 64, 48, 80, Timestamp::new(0, tb)).expect("valid frame");
    assert_eq!(f.width(), 64);
    assert_eq!(f.stride(), 80);
  }

  #[test]
  fn try_new_rejects_stride_less_than_width() {
    let buf = [0u8; 64 * 48];
    let tb = Timebase::new(1, nz(1000));
    let err = LumaFrame::try_new(&buf, 64, 48, 32, Timestamp::new(0, tb)).expect_err("should fail");
    assert_eq!(
      err,
      LumaFrameError::StrideTooSmall {
        width: 64,
        stride: 32,
      },
    );
  }

  #[test]
  fn try_new_rejects_short_data() {
    let buf = [0u8; 10];
    let tb = Timebase::new(1, nz(1000));
    let err = LumaFrame::try_new(&buf, 64, 48, 64, Timestamp::new(0, tb)).expect_err("should fail");
    assert_eq!(
      err,
      LumaFrameError::DataTooShort {
        expected: 64 * 48,
        actual: 10,
      },
    );
  }

  #[test]
  fn luma_frame_error_display() {
    let e = LumaFrameError::StrideTooSmall {
      width: 64,
      stride: 32,
    };
    assert_eq!(format!("{e}"), "stride (32) is smaller than width (64)");
  }

  #[test]
  fn rgb_frame_basic() {
    let buf = [0u8; 4 * 3 * 2];
    let tb = Timebase::new(1, nz(1000));
    let f = RgbFrame::new(&buf, 4, 2, 12, Timestamp::new(0, tb));
    assert_eq!(f.width(), 4);
    assert_eq!(f.height(), 2);
    assert_eq!(f.stride(), 12);
    assert_eq!(f.data().len(), 24);
  }

  #[test]
  fn rgb_frame_with_padding() {
    // 4-pixel row = 12 bytes of pixel data + 4 bytes of alignment padding.
    let buf = [0u8; 16 * 2];
    let tb = Timebase::new(1, nz(1000));
    let f = RgbFrame::new(&buf, 4, 2, 16, Timestamp::new(0, tb));
    assert_eq!(f.stride(), 16);
  }

  #[test]
  fn try_new_rgb_rejects_stride_less_than_width_times_3() {
    let buf = [0u8; 12 * 2];
    let tb = Timebase::new(1, nz(1000));
    let err =
      RgbFrame::try_new(&buf, 4, 2, 8, Timestamp::new(0, tb)).expect_err("stride 8 < 4*3 = 12");
    assert_eq!(
      err,
      RgbFrameError::StrideTooSmall {
        width: 4,
        stride: 8,
        min_stride: 12,
      },
    );
  }

  #[test]
  fn try_new_rgb_rejects_short_data() {
    let buf = [0u8; 10];
    let tb = Timebase::new(1, nz(1000));
    let err = RgbFrame::try_new(&buf, 4, 2, 12, Timestamp::new(0, tb)).expect_err("should fail");
    assert_eq!(
      err,
      RgbFrameError::DataTooShort {
        expected: 24,
        actual: 10,
      },
    );
  }

  #[test]
  #[should_panic(expected = "invalid RgbFrame")]
  fn rgb_frame_new_panics_on_invalid() {
    let buf = [0u8; 10];
    let tb = Timebase::new(1, nz(1000));
    let _ = RgbFrame::new(&buf, 4, 2, 12, Timestamp::new(0, tb));
  }

  #[test]
  fn rgb_frame_try_new_rejects_width_times_three_overflow() {
    // width * BYTES_PER_PIXEL (3) overflows u32 when width > u32::MAX / 3.
    let buf = [0u8; 0];
    let tb = Timebase::new(1, nz(1000));
    let bad_w = u32::MAX / 3 + 1;
    let err = RgbFrame::try_new(&buf, bad_w, 1, u32::MAX, Timestamp::new(0, tb))
      .expect_err("width*3 should overflow");
    assert_eq!(err, RgbFrameError::WidthOverflow { width: bad_w });
  }

  // -------------------------------------------------------------------------
  // HsvFrame
  // -------------------------------------------------------------------------

  #[test]
  fn hsv_frame_basic_accessors() {
    let h = vec![10u8; 64 * 48];
    let s = vec![20u8; 64 * 48];
    let v = vec![30u8; 64 * 48];
    let tb = Timebase::new(1, nz(1000));
    let ts = Timestamp::new(42, tb);
    let f = HsvFrame::new(&h, &s, &v, 64, 48, 64, ts);

    assert_eq!(f.width(), 64);
    assert_eq!(f.height(), 48);
    assert_eq!(f.stride(), 64);
    assert_eq!(f.timestamp(), ts);
    assert_eq!(f.hue().len(), 64 * 48);
    assert_eq!(f.saturation().len(), 64 * 48);
    assert_eq!(f.value().len(), 64 * 48);
    assert_eq!(f.hue()[0], 10);
    assert_eq!(f.saturation()[0], 20);
    assert_eq!(f.value()[0], 30);
  }

  #[test]
  fn hsv_frame_try_new_rejects_stride_less_than_width() {
    let h = vec![0u8; 16];
    let tb = Timebase::new(1, nz(1000));
    let err =
      HsvFrame::try_new(&h, &h, &h, 64, 1, 32, Timestamp::new(0, tb)).expect_err("should fail");
    assert_eq!(
      err,
      HsvFrameError::StrideTooSmall {
        width: 64,
        stride: 32
      }
    );
  }

  #[test]
  fn hsv_frame_try_new_reports_which_plane_is_short() {
    let full = vec![0u8; 64 * 48];
    let short = vec![0u8; 10];
    let tb = Timebase::new(1, nz(1000));
    let ts = Timestamp::new(0, tb);

    // H short → reports Hue.
    let err = HsvFrame::try_new(&short, &full, &full, 64, 48, 64, ts).expect_err("h too short");
    assert_eq!(
      err,
      HsvFrameError::PlaneTooShort {
        plane: HsvPlane::Hue,
        expected: 64 * 48,
        actual: 10,
      },
    );

    // S short → reports Saturation.
    let err = HsvFrame::try_new(&full, &short, &full, 64, 48, 64, ts).expect_err("s too short");
    assert_eq!(
      err,
      HsvFrameError::PlaneTooShort {
        plane: HsvPlane::Saturation,
        expected: 64 * 48,
        actual: 10,
      },
    );

    // V short → reports Value.
    let err = HsvFrame::try_new(&full, &full, &short, 64, 48, 64, ts).expect_err("v too short");
    assert_eq!(
      err,
      HsvFrameError::PlaneTooShort {
        plane: HsvPlane::Value,
        expected: 64 * 48,
        actual: 10,
      },
    );
  }

  #[test]
  #[should_panic(expected = "invalid HsvFrame")]
  fn hsv_frame_new_panics_on_invalid() {
    let h = vec![0u8; 10];
    let tb = Timebase::new(1, nz(1000));
    let _ = HsvFrame::new(&h, &h, &h, 64, 48, 64, Timestamp::new(0, tb));
  }

  #[test]
  fn hsv_plane_display_and_as_str() {
    assert_eq!(HsvPlane::Hue.as_str(), "hue");
    assert_eq!(HsvPlane::Saturation.as_str(), "saturation");
    assert_eq!(HsvPlane::Value.as_str(), "value");
    assert_eq!(format!("{}", HsvPlane::Hue), "hue");
    assert_eq!(format!("{}", HsvPlane::Saturation), "saturation");
    assert_eq!(format!("{}", HsvPlane::Value), "value");
  }

  #[test]
  fn hsv_frame_error_display_variants() {
    let e = HsvFrameError::StrideTooSmall {
      width: 10,
      stride: 5,
    };
    assert!(format!("{e}").contains("smaller than width"));
    let e = HsvFrameError::PlaneTooShort {
      plane: HsvPlane::Saturation,
      expected: 100,
      actual: 50,
    };
    let s = format!("{e}");
    assert!(s.contains("saturation"));
    assert!(s.contains("100"));
    assert!(s.contains("50"));
  }

  #[test]
  fn frame_error_displays_include_key_fields() {
    // RgbFrameError::{StrideTooSmall, DataTooShort, DimensionsOverflow}
    let e = RgbFrameError::StrideTooSmall {
      width: 4,
      stride: 8,
      min_stride: 12,
    };
    assert!(format!("{e}").contains("12"));
    let e = RgbFrameError::DataTooShort {
      expected: 24,
      actual: 10,
    };
    assert!(format!("{e}").contains("24"));
    let e = RgbFrameError::DimensionsOverflow {
      stride: 1,
      height: 1,
    };
    assert!(format!("{e}").contains("overflow"));

    // LumaFrameError::{DataTooShort, DimensionsOverflow}
    let e = LumaFrameError::DataTooShort {
      expected: 24,
      actual: 10,
    };
    assert!(format!("{e}").contains("24"));
    let e = LumaFrameError::DimensionsOverflow {
      stride: 1,
      height: 1,
    };
    assert!(format!("{e}").contains("overflow"));

    // HsvFrameError::DimensionsOverflow
    let e = HsvFrameError::DimensionsOverflow {
      stride: 1,
      height: 1,
    };
    assert!(format!("{e}").contains("overflow"));
  }
}
