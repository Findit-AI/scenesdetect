use core::{
  cmp::Ordering,
  hash::{Hash, Hasher},
  num::NonZeroU32,
  time::Duration,
};

/// A media timebase represented as a rational number: numerator over non-zero denominator.
///
/// Typical values: `1/1000` for millisecond PTS, `1/90000` for MPEG-TS,
/// `1/48000` for audio samples, `30000/1001` for NTSC video (when used as a
/// frame rate).
///
/// # Equality and ordering
///
/// Comparison is **value-based**: `1/2` equals `2/4`, and `1/3 < 2/3 < 1/1`.
/// [`Hash`] hashes the reduced (lowest-terms) form, so equal rationals hash
/// the same. Cross-multiplication uses `u64` intermediates — exact for any
/// `u32` numerator / denominator.
#[derive(Debug, Clone, Copy)]
pub struct Timebase {
  num: u32,
  den: NonZeroU32,
}

impl Timebase {
  /// Creates a new `Timebase` with the given numerator and non-zero denominator.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(num: u32, den: NonZeroU32) -> Self {
    Self { num, den }
  }

  /// Returns the numerator.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn num(&self) -> u32 {
    self.num
  }

  /// Returns the denominator.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn den(&self) -> NonZeroU32 {
    self.den
  }

  /// Set the value of the numerator.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_num(mut self, num: u32) -> Self {
    self.set_num(num);
    self
  }

  /// Set the value of the denominator.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_den(mut self, den: NonZeroU32) -> Self {
    self.set_den(den);
    self
  }

  /// Set the value of the numerator in place.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_num(&mut self, num: u32) -> &mut Self {
    self.num = num;
    self
  }

  /// Set the value of the denominator in place.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_den(&mut self, den: NonZeroU32) -> &mut Self {
    self.den = den;
    self
  }

  /// Rescales `pts` from timebase `from` to timebase `to`, rounding toward zero.
  ///
  /// Equivalent to FFmpeg's `av_rescale_q`. Uses a 128-bit intermediate to
  /// avoid overflow for typical video PTS ranges.
  ///
  /// # Panics
  ///
  /// Panics if `to.num() == 0` (division by zero).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn rescale_pts(pts: i64, from: Self, to: Self) -> i64 {
    // pts * (from.num / from.den) / (to.num / to.den)
    // = pts * from.num * to.den / (from.den * to.num)
    let numerator = (pts as i128) * (from.num as i128) * (to.den.get() as i128);
    let denominator = (from.den.get() as i128) * (to.num as i128);
    (numerator / denominator) as i64
  }

  /// Rescales `pts` from this timebase to `to`, rounding toward zero.
  ///
  /// Method form of [`Self::rescale_pts`]: `self` is the source timebase.
  ///
  /// # Panics
  ///
  /// Panics if `to.num() == 0` (division by zero).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn rescale(&self, pts: i64, to: Self) -> i64 {
    Self::rescale_pts(pts, *self, to)
  }

  /// Treats `self` as a frame rate (frames per second) and returns the
  /// [`Duration`] corresponding to `frames` frames.
  ///
  /// Examples:
  /// - 30 fps: `Timebase::new(30, nz(1)).frames_to_duration(15)` → 500 ms
  /// - NTSC: `Timebase::new(30000, nz(1001)).frames_to_duration(30000)` → 1001 ms
  ///
  /// Note that "frame rate" and "PTS timebase" are conceptually *different*
  /// rationals even though both are represented as [`Timebase`]. A 30 fps
  /// stream typically has PTS timebase `1/30` (seconds per unit) and frame
  /// rate `30/1` (frames per second) — they are reciprocals.
  ///
  /// # Panics
  ///
  /// Panics if `self.num() == 0` (division by zero).
  pub const fn frames_to_duration(&self, frames: u32) -> Duration {
    // frames / (num/den) seconds = frames * den / num seconds
    let num = self.num as u128;
    let den = self.den.get() as u128;
    assert!(num != 0, "frame rate numerator must be non-zero");
    let total_ns = (frames as u128) * den * 1_000_000_000 / num;
    let secs = (total_ns / 1_000_000_000) as u64;
    let nanos = (total_ns % 1_000_000_000) as u32;
    Duration::new(secs, nanos)
  }

  /// Converts a [`Duration`] into the number of PTS units this timebase
  /// represents, rounding toward zero.
  ///
  /// Inverse of "multiplying a PTS value by this timebase to get seconds".
  /// Saturates at `i64::MAX` if the duration is absurdly large for this
  /// timebase. Returns `0` if `self.num() == 0` (a degenerate timebase).
  pub const fn duration_to_pts(&self, d: Duration) -> i64 {
    let num = self.num as u128;
    if num == 0 {
      return 0;
    }
    let den = self.den.get() as u128;
    // pts_units = duration_ns * den / (num * 1e9)
    let ns = d.as_nanos();
    let pts = ns * den / (num * 1_000_000_000);
    if pts > i64::MAX as u128 {
      i64::MAX
    } else {
      pts as i64
    }
  }
}

impl PartialEq for Timebase {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn eq(&self, other: &Self) -> bool {
    // a.num * b.den == b.num * a.den (cross-multiply; u32 * u32 fits in u64)
    (self.num as u64) * (other.den.get() as u64) == (other.num as u64) * (self.den.get() as u64)
  }
}
impl Eq for Timebase {}

impl Hash for Timebase {
  fn hash<H: Hasher>(&self, state: &mut H) {
    let d = self.den.get();
    // gcd(num, d) ≥ 1 because d ≥ 1 (NonZeroU32).
    let g = gcd_u32(self.num, d);
    (self.num / g).hash(state);
    (d / g).hash(state);
  }
}

impl Ord for Timebase {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn cmp(&self, other: &Self) -> Ordering {
    let lhs = (self.num as u64) * (other.den.get() as u64);
    let rhs = (other.num as u64) * (self.den.get() as u64);
    lhs.cmp(&rhs)
  }
}
impl PartialOrd for Timebase {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    Some(self.cmp(other))
  }
}

/// A presentation timestamp, expressed as a PTS value in units of an associated [`Timebase`].
///
/// # Equality and ordering
///
/// Comparison is **value-based** (same instant compares equal even across
/// different timebases): `Timestamp(1000, 1/1000)` equals
/// `Timestamp(90_000, 1/90_000)`. [`Hash`] hashes the reduced-form rational
/// instant `(pts · num, den)`, so equal timestamps hash the same.
///
/// Cross-timebase comparisons use 128-bit cross-multiplication — no division,
/// no rounding error. Same-timebase comparisons take a fast path on `pts`.
#[derive(Debug, Clone, Copy)]
pub struct Timestamp {
  pts: i64,
  timebase: Timebase,
}

impl Timestamp {
  /// Creates a new `Timestamp` with the given PTS and timebase.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(pts: i64, timebase: Timebase) -> Self {
    Self { pts, timebase }
  }

  /// Returns the presentation timestamp, in units of [`Self::timebase`].
  ///
  /// To obtain a [`Duration`], use [`Self::duration_since`] against a reference
  /// timestamp, or rescale via [`Self::rescale_to`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn pts(&self) -> i64 {
    self.pts
  }

  /// Returns the timebase of the timestamp.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn timebase(&self) -> Timebase {
    self.timebase
  }

  /// Set the value of the presentation timestamp.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_pts(mut self, pts: i64) -> Self {
    self.set_pts(pts);
    self
  }

  /// Set the value of the presentation timestamp in place.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_pts(&mut self, pts: i64) -> &mut Self {
    self.pts = pts;
    self
  }

  /// Returns a new `Timestamp` representing the same instant in a different timebase.
  ///
  /// Rounds toward zero via [`Timebase::rescale_pts`]; round-tripping through a
  /// coarser timebase can lose precision.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn rescale_to(self, target: Timebase) -> Self {
    Self {
      pts: self.timebase.rescale(self.pts, target),
      timebase: target,
    }
  }

  /// Returns a new [`Timestamp`] representing this instant shifted backward
  /// by `d`, in the same timebase. Saturates at `i64::MIN` if the subtraction
  /// would underflow (pathological for real video).
  ///
  /// Useful for "virtual past" seeding: e.g., initializing a warmup-filter
  /// state to `ts - min_duration` so the first detected cut can fire
  /// immediately.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn saturating_sub_duration(self, d: Duration) -> Self {
    let units = self.timebase.duration_to_pts(d);
    Self::new(self.pts.saturating_sub(units), self.timebase)
  }

  /// `const fn` form of [`Ord::cmp`]. Compares two timestamps by the instant
  /// they represent, rescaling if timebases differ.
  ///
  /// Uses a 128-bit cross-multiply for the mixed-timebase case; no division,
  /// so no rounding error. Same-timebase comparisons take a direct fast path.
  pub const fn cmp_semantic(&self, other: &Self) -> Ordering {
    if self.timebase.num == other.timebase.num
      && self.timebase.den.get() == other.timebase.den.get()
    {
      return if self.pts < other.pts {
        Ordering::Less
      } else if self.pts > other.pts {
        Ordering::Greater
      } else {
        Ordering::Equal
      };
    }
    // self.pts * self.num / self.den  vs  other.pts * other.num / other.den
    //   ⇔ self.pts * self.num * other.den  vs  other.pts * other.num * self.den
    let lhs = (self.pts as i128) * (self.timebase.num as i128) * (other.timebase.den.get() as i128);
    let rhs =
      (other.pts as i128) * (other.timebase.num as i128) * (self.timebase.den.get() as i128);
    if lhs < rhs {
      Ordering::Less
    } else if lhs > rhs {
      Ordering::Greater
    } else {
      Ordering::Equal
    }
  }

  /// Returns the elapsed [`Duration`] from `earlier` to `self`, or `None` if
  /// `earlier` is after `self`.
  ///
  /// Works across different timebases. Computes the difference in nanoseconds
  /// via 128-bit intermediates; for realistic video PTS ranges this is exact,
  /// but pathological inputs may saturate.
  pub const fn duration_since(&self, earlier: &Self) -> Option<Duration> {
    // nanos = pts * tb.num * 1_000_000_000 / tb.den
    const NS_PER_SEC: i128 = 1_000_000_000;
    let self_ns = (self.pts as i128) * (self.timebase.num as i128) * NS_PER_SEC
      / (self.timebase.den.get() as i128);
    let earlier_ns = (earlier.pts as i128) * (earlier.timebase.num as i128) * NS_PER_SEC
      / (earlier.timebase.den.get() as i128);
    let diff = self_ns - earlier_ns;
    if diff < 0 {
      return None;
    }
    let secs = (diff / NS_PER_SEC) as u64;
    let nanos = (diff % NS_PER_SEC) as u32;
    Some(Duration::new(secs, nanos))
  }
}

impl PartialEq for Timestamp {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn eq(&self, other: &Self) -> bool {
    self.cmp_semantic(other).is_eq()
  }
}
impl Eq for Timestamp {}

impl Hash for Timestamp {
  fn hash<H: Hasher>(&self, state: &mut H) {
    // Canonical representation: instant as reduced rational (pts * num, den).
    let n: i128 = (self.pts as i128) * (self.timebase.num as i128);
    let d: u128 = self.timebase.den.get() as u128;
    // gcd operates on magnitudes; denominator stays positive. gcd ≥ 1 since d ≥ 1.
    let g = gcd_u128(n.unsigned_abs(), d) as i128;
    let rn = n / g;
    let rd = (d as i128) / g;
    rn.hash(state);
    rd.hash(state);
  }
}

impl Ord for Timestamp {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn cmp(&self, other: &Self) -> Ordering {
    self.cmp_semantic(other)
  }
}
impl PartialOrd for Timestamp {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    Some(self.cmp(other))
  }
}

/// A half-open time range `[start, end)` in a given [`Timebase`].
///
/// Represents the extent of a detected event — for example, the
/// fade-out→fade-in duration exposed by
/// [`crate::threshold::Detector::last_fade_range`]. When `start == end`,
/// the range is degenerate (an instant); see [`Self::instant`].
///
/// Both endpoints share the same [`Timebase`]. To compare ranges across
/// different timebases, rescale one of them first (e.g., by calling
/// [`Timestamp::rescale_to`] on each endpoint).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TimeRange {
  start: i64,
  end: i64,
  timebase: Timebase,
}

impl TimeRange {
  /// Creates a new `TimeRange` with the given start/end PTS and shared timebase.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(start: i64, end: i64, timebase: Timebase) -> Self {
    Self {
      start,
      end,
      timebase,
    }
  }

  /// Creates a degenerate (instant) range where `start == end == ts.pts()`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn instant(ts: Timestamp) -> Self {
    Self {
      start: ts.pts(),
      end: ts.pts(),
      timebase: ts.timebase(),
    }
  }

  /// Returns the start PTS in the range's timebase units.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn start_pts(&self) -> i64 {
    self.start
  }

  /// Returns the end PTS in the range's timebase units.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn end_pts(&self) -> i64 {
    self.end
  }

  /// Returns the shared timebase.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn timebase(&self) -> Timebase {
    self.timebase
  }

  /// Returns the start as a [`Timestamp`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn start(&self) -> Timestamp {
    Timestamp::new(self.start, self.timebase)
  }

  /// Returns the end as a [`Timestamp`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn end(&self) -> Timestamp {
    Timestamp::new(self.end, self.timebase)
  }

  /// Sets the start PTS.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_start(mut self, val: i64) -> Self {
    self.start = val;
    self
  }

  /// Sets the start PTS in place.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_start(&mut self, val: i64) -> &mut Self {
    self.start = val;
    self
  }

  /// Sets the end PTS.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_end(mut self, val: i64) -> Self {
    self.end = val;
    self
  }

  /// Sets the end PTS in place.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_end(&mut self, val: i64) -> &mut Self {
    self.end = val;
    self
  }

  /// Returns `true` if `start == end` (a degenerate instant range).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn is_instant(&self) -> bool {
    self.start == self.end
  }

  /// Returns the elapsed [`Duration`] from `start` to `end`, or `None` if
  /// `end` is before `start`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn duration(&self) -> Option<Duration> {
    self.end().duration_since(&self.start())
  }

  /// Linearly interpolates between `start` and `end`: `t = 0.0` returns
  /// `start`, `t = 1.0` returns `end`, `t = 0.5` the midpoint. `t` is
  /// clamped to `[0.0, 1.0]`. Rounds toward zero.
  ///
  /// Use this to map an old-style bias value `b ∈ [-1, 1]` onto the range:
  /// `range.interpolate((b + 1.0) * 0.5)`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn interpolate(&self, t: f64) -> Timestamp {
    let t = t.clamp(0.0, 1.0);
    let delta = self.end.saturating_sub(self.start);
    let offset = (delta as f64 * t) as i64;
    Timestamp::new(self.start.saturating_add(offset), self.timebase)
  }
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
      None => return Err(RgbFrameError::DimensionsOverflow { stride, height }),
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
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
  /// `width * 3` or `stride * height` overflowed `usize` (can only happen
  /// on 32-bit targets with very large frames).
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HsvPlane {
  /// Hue plane.
  Hue,
  /// Saturation plane.
  Saturation,
  /// Value (brightness) plane.
  Value,
}

impl core::fmt::Display for HsvPlane {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    match self {
      Self::Hue => f.write_str("hue"),
      Self::Saturation => f.write_str("saturation"),
      Self::Value => f.write_str("value"),
    }
  }
}

/// Error returned by [`HsvFrame::try_new`] when the planes are inconsistent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
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

const fn gcd_u32(mut a: u32, mut b: u32) -> u32 {
  while b != 0 {
    let t = b;
    b = a % b;
    a = t;
  }
  a
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn gcd_u128(mut a: u128, mut b: u128) -> u128 {
  while b != 0 {
    let t = b;
    b = a % b;
    a = t;
  }
  a
}

#[cfg(test)]
mod tests {
  use super::*;

  const fn nz(n: u32) -> NonZeroU32 {
    match NonZeroU32::new(n) {
      Some(v) => v,
      None => panic!("zero"),
    }
  }

  fn hash_of<T: Hash>(v: &T) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut h = DefaultHasher::new();
    v.hash(&mut h);
    h.finish()
  }

  #[test]
  fn rescale_identity() {
    let tb = Timebase::new(1, nz(1000));
    assert_eq!(Timebase::rescale_pts(42, tb, tb), 42);
    assert_eq!(tb.rescale(42, tb), 42);
  }

  #[test]
  fn rescale_between_timebases() {
    let ms = Timebase::new(1, nz(1000));
    let mpeg = Timebase::new(1, nz(90_000));
    assert_eq!(Timebase::rescale_pts(1000, ms, mpeg), 90_000);
    assert_eq!(ms.rescale(1000, mpeg), 90_000);
    assert_eq!(mpeg.rescale(90_000, ms), 1000);
  }

  #[test]
  fn rescale_rounds_toward_zero() {
    let from = Timebase::new(1, nz(1000));
    let to = Timebase::new(1, nz(3));
    assert_eq!(from.rescale(1, to), 0);
    assert_eq!(from.rescale(-1, to), 0);
  }

  #[test]
  fn timebase_eq_is_semantic() {
    // 1/2 == 2/4 == 3/6
    let a = Timebase::new(1, nz(2));
    let b = Timebase::new(2, nz(4));
    let c = Timebase::new(3, nz(6));
    assert_eq!(a, b);
    assert_eq!(b, c);
    assert_eq!(a, c);
    // 1/2 != 1/3
    let d = Timebase::new(1, nz(3));
    assert_ne!(a, d);
  }

  #[test]
  fn timebase_hash_matches_eq() {
    let a = Timebase::new(1, nz(2));
    let b = Timebase::new(2, nz(4));
    let c = Timebase::new(3, nz(6));
    assert_eq!(hash_of(&a), hash_of(&b));
    assert_eq!(hash_of(&b), hash_of(&c));
  }

  #[test]
  fn timebase_ord_is_numeric() {
    let third = Timebase::new(1, nz(3));
    let half = Timebase::new(1, nz(2));
    let two_thirds = Timebase::new(2, nz(3));
    let one = Timebase::new(1, nz(1));
    assert!(third < half);
    assert!(half < two_thirds);
    assert!(two_thirds < one);
    // Structural lex order would have reported (1, 1) < (1, 3); verify it doesn't.
    assert!(one > third);
  }

  #[test]
  fn timebase_num_zero() {
    // 0/3 == 0/5, and both compare less than anything positive.
    let a = Timebase::new(0, nz(3));
    let b = Timebase::new(0, nz(5));
    assert_eq!(a, b);
    assert_eq!(hash_of(&a), hash_of(&b));
    assert!(a < Timebase::new(1, nz(1_000_000)));
  }

  #[test]
  fn timestamp_cmp_same_timebase() {
    let tb = Timebase::new(1, nz(1000));
    let a = Timestamp::new(100, tb);
    let b = Timestamp::new(200, tb);
    assert!(a < b);
    assert!(b > a);
    assert_eq!(a, a);
    assert_eq!(a.cmp(&b), Ordering::Less);
  }

  #[test]
  fn timestamp_cmp_cross_timebase() {
    let a = Timestamp::new(1000, Timebase::new(1, nz(1000)));
    let b = Timestamp::new(90_000, Timebase::new(1, nz(90_000)));
    assert_eq!(a, b);
    assert_eq!(a.cmp(&b), Ordering::Equal);

    let c = Timestamp::new(500, Timebase::new(1, nz(1000)));
    assert!(c < a);
    assert!(a > c);
  }

  #[test]
  fn timestamp_hash_matches_semantic_eq() {
    let a = Timestamp::new(1000, Timebase::new(1, nz(1000)));
    let b = Timestamp::new(90_000, Timebase::new(1, nz(90_000)));
    let c = Timestamp::new(2000, Timebase::new(1, nz(2000))); // also 1.0s
    assert_eq!(a, b);
    assert_eq!(hash_of(&a), hash_of(&b));
    assert_eq!(hash_of(&a), hash_of(&c));
  }

  #[test]
  fn timestamp_hash_negative_pts() {
    // Pre-roll / edit list scenarios: -500 ms should equal -45_000 @ 1/90_000.
    let a = Timestamp::new(-500, Timebase::new(1, nz(1000)));
    let b = Timestamp::new(-45_000, Timebase::new(1, nz(90_000)));
    assert_eq!(a, b);
    assert_eq!(hash_of(&a), hash_of(&b));
  }

  #[test]
  fn rescale_to_preserves_instant() {
    let ms = Timebase::new(1, nz(1000));
    let mpeg = Timebase::new(1, nz(90_000));
    let a = Timestamp::new(1000, ms);
    let b = a.rescale_to(mpeg);
    assert_eq!(b.pts(), 90_000);
    assert_eq!(b.timebase(), mpeg);
    assert_eq!(a, b);
  }

  #[test]
  fn duration_since_same_timebase() {
    let tb = Timebase::new(1, nz(1000));
    let a = Timestamp::new(1500, tb);
    let b = Timestamp::new(500, tb);
    assert_eq!(a.duration_since(&b), Some(Duration::from_millis(1000)));
    assert_eq!(b.duration_since(&a), None);
  }

  #[test]
  fn duration_since_cross_timebase() {
    let a = Timestamp::new(1000, Timebase::new(1, nz(1000)));
    let b = Timestamp::new(45_000, Timebase::new(1, nz(90_000)));
    assert_eq!(a.duration_since(&b), Some(Duration::from_millis(500)));
  }

  #[test]
  fn frames_to_duration_integer_fps() {
    let fps30 = Timebase::new(30, nz(1));
    assert_eq!(fps30.frames_to_duration(15), Duration::from_millis(500));
    assert_eq!(fps30.frames_to_duration(30), Duration::from_secs(1));
    assert_eq!(fps30.frames_to_duration(0), Duration::ZERO);
  }

  #[test]
  fn frames_to_duration_ntsc() {
    // 30000 frames @ 30000/1001 fps = exactly 1001 seconds.
    let ntsc = Timebase::new(30_000, nz(1001));
    assert_eq!(ntsc.frames_to_duration(30_000), Duration::from_secs(1001));
    // 15 frames at NTSC ≈ 500.5 ms.
    assert_eq!(
      ntsc.frames_to_duration(15),
      Duration::from_nanos(500_500_000),
    );
  }

  #[test]
  fn time_range_basic() {
    let tb = Timebase::new(1, nz(1000));
    let r = TimeRange::new(100, 500, tb);
    assert_eq!(r.start_pts(), 100);
    assert_eq!(r.end_pts(), 500);
    assert_eq!(r.timebase(), tb);
    assert_eq!(r.start(), Timestamp::new(100, tb));
    assert_eq!(r.end(), Timestamp::new(500, tb));
    assert!(!r.is_instant());
    assert_eq!(r.duration(), Some(Duration::from_millis(400)));
    // Interpolate: t=0 → start, t=1 → end, t=0.5 → midpoint.
    assert_eq!(r.interpolate(0.0).pts(), 100);
    assert_eq!(r.interpolate(1.0).pts(), 500);
    assert_eq!(r.interpolate(0.5).pts(), 300);
    // Out-of-range t is clamped.
    assert_eq!(r.interpolate(-1.0).pts(), 100);
    assert_eq!(r.interpolate(2.0).pts(), 500);
  }

  #[test]
  fn time_range_instant() {
    let tb = Timebase::new(1, nz(1000));
    let ts = Timestamp::new(123, tb);
    let r = TimeRange::instant(ts);
    assert!(r.is_instant());
    assert_eq!(r.start_pts(), 123);
    assert_eq!(r.end_pts(), 123);
    assert_eq!(r.duration(), Some(Duration::ZERO));
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
}
