//! Keyframe selection state machine.
//!
//! Buffers per-frame [`FrameScore`]s as they stream in, then — when the
//! caller confirms a shot boundary — partitions the buffered scores into
//! N time-uniform buckets and emits the sharpest frame per bucket.
//!
//! # Pipeline
//!
//! 1. [`Detector::observe`] — called on every scored frame. O(1) append
//!    to an internal [`VecDeque`].
//! 2. [`Detector::finalize_shot`] — called once the application has
//!    confirmed a shot boundary (typically from a merged stream of cuts
//!    produced by the scene-detector stack). Drains every buffered entry
//!    with `ts ∈ [range.start, range.end)`, buckets them, and returns
//!    the winning timestamps in PTS order. Entries strictly older than
//!    `range.start` are discarded (they belonged to an earlier shot that
//!    was never finalised).
//! 3. [`Detector::clear`] — resets the buffer between videos.
//!
//! # Bucketing
//!
//! `N = clamp(ceil(duration / target_interval), 1, max_frames_per_shot)`
//!
//! Buckets are equal-width time slices of the shot. The first and last
//! buckets are shrunk inward by `margin_ratio · duration` to protect
//! against dissolve tails and ±2-frame scene-detector slop. If the
//! margin eats the entire bucket (large `margin_ratio` on a shot with
//! only one bucket) the detector falls back to the un-shrunk range so
//! the bucket still has a chance to emit.
//!
//! # Strict vs. fallback selection
//!
//! Per bucket, the detector tracks two running argmaxes:
//!
//! - **strict**: argmax over frames passing every hard gate (luma mean,
//!   saturation / luma variance AND-gate, clipping, minimum sharpness).
//! - **fallback**: argmax over all frames in the bucket, regardless of
//!   gate outcome.
//!
//! The emitted timestamp is the strict winner when non-empty; otherwise
//! the fallback winner. This mirrors the reference Python algorithm's
//! "prefer a degraded frame over a gap in temporal coverage" policy —
//! VLMs downstream benefit more from regular temporal sampling than from
//! the occasional missing keyframe.

use core::cmp::Ordering;
use core::time::Duration;

use std::collections::VecDeque;
use std::vec::Vec;

use crate::frame::{TimeRange, Timestamp};
use crate::keyframe::score::FrameScore;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// ---- Options ---------------------------------------------------------------

/// Tuning knobs for the selection state machine.
///
/// Default values are calibrated for 256-px longest-side preprocessed
/// input (the default of [`crate::keyframe::preprocess::Downscaler`]).
/// Changing the downscale dimension invalidates the sharpness and
/// variance thresholds — keep them in sync.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Options {
  target_interval: Duration,
  max_frames_per_shot: u32,
  margin_ratio: f64,
  min_sharpness: f32,
  black_mean_threshold: u8,
  bright_mean_threshold: u8,
  luma_variance_threshold: f32,
  sat_variance_threshold: f32,
  max_clipping: f32,
}

impl Default for Options {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn default() -> Self {
    Self {
      target_interval: Duration::from_secs(4),
      max_frames_per_shot: 16,
      margin_ratio: 0.02,
      min_sharpness: 100.0,
      black_mean_threshold: 15,
      bright_mean_threshold: 240,
      luma_variance_threshold: 5.0,
      sat_variance_threshold: 3.0,
      max_clipping: 0.5,
    }
  }
}

impl Options {
  /// Creates a new [`Options`] matching [`Options::default`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new() -> Self {
    Self::default()
  }

  /// Target interval between keyframes. Drives the bucket count via
  /// `ceil(shot_duration / target_interval)`.
  ///
  /// # Panics
  /// When `d` is [`Duration::ZERO`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_target_interval(mut self, d: Duration) -> Self {
    assert!(!d.is_zero(), "target_interval must be > 0");
    self.target_interval = d;
    self
  }

  /// Upper bound on the number of keyframes emitted per shot.
  ///
  /// # Panics
  /// When `n == 0`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_max_frames_per_shot(mut self, n: u32) -> Self {
    assert!(n > 0, "max_frames_per_shot must be > 0");
    self.max_frames_per_shot = n;
    self
  }

  /// Fraction of shot duration to shrink away from the first and last
  /// buckets.
  ///
  /// # Panics
  /// When outside `[0.0, 0.5)`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_margin_ratio(mut self, r: f64) -> Self {
    assert!(
      (0.0..0.5).contains(&r),
      "margin_ratio must be in [0.0, 0.5)"
    );
    self.margin_ratio = r;
    self
  }

  /// Minimum Tenengrad sharpness for the strict pass.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_sharpness(mut self, s: f32) -> Self {
    self.min_sharpness = s;
    self
  }

  /// Luma-mean floor — frames below this are flagged too-dark.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_black_mean_threshold(mut self, t: u8) -> Self {
    self.black_mean_threshold = t;
    self
  }

  /// Luma-mean ceiling — frames above this are flagged overexposed.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_bright_mean_threshold(mut self, t: u8) -> Self {
    self.bright_mean_threshold = t;
    self
  }

  /// Luma-variance floor. AND-gated with
  /// [`Self::with_sat_variance_threshold`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_luma_variance_threshold(mut self, t: f32) -> Self {
    self.luma_variance_threshold = t;
    self
  }

  /// Saturation-variance floor. AND-gated with
  /// [`Self::with_luma_variance_threshold`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_sat_variance_threshold(mut self, t: f32) -> Self {
    self.sat_variance_threshold = t;
    self
  }

  /// Maximum tolerated clipping ratio.
  ///
  /// # Panics
  /// When outside `[0.0, 1.0]`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_max_clipping(mut self, c: f32) -> Self {
    assert!(
      (0.0..=1.0).contains(&c),
      "max_clipping must be in [0.0, 1.0]"
    );
    self.max_clipping = c;
    self
  }

  /// Target inter-keyframe interval (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn target_interval(&self) -> Duration {
    self.target_interval
  }
  /// Maximum keyframes per shot (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn max_frames_per_shot(&self) -> u32 {
    self.max_frames_per_shot
  }
  /// First/last-bucket margin fraction (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn margin_ratio(&self) -> f64 {
    self.margin_ratio
  }
  /// Strict-pass sharpness floor (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_sharpness(&self) -> f32 {
    self.min_sharpness
  }
  /// Black-frame luma-mean threshold (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn black_mean_threshold(&self) -> u8 {
    self.black_mean_threshold
  }
  /// Overexposed luma-mean threshold (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn bright_mean_threshold(&self) -> u8 {
    self.bright_mean_threshold
  }
  /// Luma-variance threshold (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn luma_variance_threshold(&self) -> f32 {
    self.luma_variance_threshold
  }
  /// Saturation-variance threshold (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn sat_variance_threshold(&self) -> f32 {
    self.sat_variance_threshold
  }
  /// Max clipping ratio (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn max_clipping(&self) -> f32 {
    self.max_clipping
  }
}

// ---- Detector --------------------------------------------------------------

/// The keyframe selection state machine.
#[derive(Debug, Clone)]
pub struct Detector {
  opts: Options,
  buffer: VecDeque<(Timestamp, FrameScore)>,
}

impl Detector {
  /// Creates a new detector with the supplied options.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new(opts: Options) -> Self {
    Self {
      opts,
      buffer: VecDeque::new(),
    }
  }

  /// Returns the detector's current options.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn options(&self) -> &Options {
    &self.opts
  }

  /// Number of scored frames currently buffered (awaiting finalization).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn buffered(&self) -> usize {
    self.buffer.len()
  }

  /// Appends a scored frame to the buffer. Callers must feed frames in
  /// non-decreasing PTS order.
  ///
  /// In debug builds, an out-of-order call panics via `debug_assert!`.
  /// In release builds the precondition is not checked; an out-of-order
  /// entry reaches the bucket-walker in [`Self::finalize_shot`] and is
  /// silently dropped (either because it predates the cursor's current
  /// bucket start, or because it lands in a gap between buckets). This
  /// is the intended failure mode — keyframe selection prefers to
  /// tolerate minor ordering slop over aborting mid-stream — but if
  /// you care about catching decoder bugs, run with debug assertions
  /// enabled.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn observe(&mut self, ts: Timestamp, score: FrameScore) {
    debug_assert!(
      self
        .buffer
        .back()
        .is_none_or(|(prev, _)| prev.cmp_semantic(&ts) != Ordering::Greater),
      "observe() frames must arrive in non-decreasing PTS order"
    );
    self.buffer.push_back((ts, score));
  }

  /// A shot boundary has been confirmed. Drains every buffered entry
  /// whose timestamp lies in `[range.start(), range.end())`, buckets
  /// them, and returns the winning timestamp per bucket in PTS order.
  ///
  /// Entries strictly older than `range.start()` are silently discarded
  /// (they belonged to an earlier shot that was never finalized, or
  /// were observed before the first shot began).
  ///
  /// A degenerate or reversed `range` (duration ≤ 0) yields an empty
  /// result and still drops stale entries.
  ///
  /// Returns an owned `Vec<Timestamp>` rather than a borrowing iterator.
  /// Size is bounded by
  /// [`Options::max_frames_per_shot`](Options::with_max_frames_per_shot) —
  /// typically ≤ 16 entries, so the allocation is small. Caller may hold
  /// the result across subsequent detector calls.
  pub fn finalize_shot(&mut self, range: TimeRange) -> Vec<Timestamp> {
    // 1. Drop stale entries (before the shot starts).
    while let Some((ts, _)) = self.buffer.front() {
      if ts.cmp_semantic(&range.start()) == Ordering::Less {
        self.buffer.pop_front();
      } else {
        break;
      }
    }

    // 2. Degenerate range (zero or negative duration) → nothing to emit.
    let duration = match range.duration() {
      Some(d) if !d.is_zero() => d,
      _ => return Vec::new(),
    };

    // 3. Compute bucket count and precompute per-bucket effective
    //    [start, end) timestamps (with first-/last-bucket margin shrink).
    let n = compute_n_buckets(duration, &self.opts);
    let bucket_ranges = compute_bucket_ranges(&range, n, self.opts.margin_ratio);

    // 4. Single linear walk across the in-range entries. Entries inside
    //    the first-bucket's pre-margin or last-bucket's post-margin zones
    //    are skipped. Track strict + fallback running argmaxes per bucket.
    let mut emits: Vec<Timestamp> = Vec::with_capacity(n);
    let mut bucket_idx = 0usize;
    let mut best_strict: Option<(Timestamp, f32)> = None;
    let mut best_any: Option<(Timestamp, f32)> = None;

    // Snapshot opts locally to avoid borrowing self while draining the
    // buffer.
    let opts = self.opts;

    while let Some((ts, score)) = self.buffer.front().copied() {
      if ts.cmp_semantic(&range.end()) != Ordering::Less {
        break; // past this shot — leave for the next finalize.
      }

      // Advance bucket cursor while the entry is past the current
      // bucket's effective end. Each advance emits the previous
      // bucket's winner.
      while bucket_idx < n && ts.cmp_semantic(&bucket_ranges[bucket_idx].1) != Ordering::Less {
        if let Some((t, _)) = best_strict.or(best_any) {
          emits.push(t);
        }
        best_strict = None;
        best_any = None;
        bucket_idx += 1;
      }

      // Consume the entry from the buffer. We're inside the shot.
      self.buffer.pop_front();

      if bucket_idx >= n {
        // Past the last bucket's effective end, but still inside the
        // shot (i.e. within the last-bucket margin zone). Drain but do
        // not score.
        continue;
      }

      // Skip entries that fall in the first bucket's pre-margin gap.
      if ts.cmp_semantic(&bucket_ranges[bucket_idx].0) == Ordering::Less {
        continue;
      }

      // Running argmax updates.
      if best_any.is_none_or(|(_, s)| sharper(score.sharpness, s)) {
        best_any = Some((ts, score.sharpness));
      }
      if !hard_gate(&score, &opts)
        && score.sharpness >= opts.min_sharpness
        && best_strict.is_none_or(|(_, s)| sharper(score.sharpness, s))
      {
        best_strict = Some((ts, score.sharpness));
      }
    }

    // Flush the last active bucket.
    if bucket_idx < n {
      if let Some((t, _)) = best_strict.or(best_any) {
        emits.push(t);
      }
    }

    emits
  }

  /// Closes out any buffered entries as a "final shot" ending at `eos`.
  ///
  /// Convenience wrapper for end-of-stream handling: forms a synthetic
  /// [`TimeRange`] that starts at the oldest buffered entry's timestamp
  /// and ends at `eos`, then calls [`Self::finalize_shot`]. Returns an
  /// empty `Vec` when the buffer is empty or when `eos` is not strictly
  /// after the earliest buffered entry.
  ///
  /// Callers that track the previous-cut timestamp themselves can
  /// equivalently call [`Self::finalize_shot`] directly with a range
  /// they construct.
  pub fn finalize_remaining(&mut self, eos: Timestamp) -> Vec<Timestamp> {
    let Some(&(first_ts, _)) = self.buffer.front() else {
      return Vec::new();
    };
    // Re-express the start timestamp in eos's timebase so the TimeRange
    // constructor (which takes raw pts + a single timebase) sees a
    // self-consistent pair.
    let tb = eos.timebase();
    let start_pts = first_ts.rescale_to(tb).pts();
    let end_pts = eos.pts();
    if end_pts <= start_pts {
      return Vec::new();
    }
    self.finalize_shot(TimeRange::new(start_pts, end_pts, tb))
  }

  /// Resets the detector's buffer. The configured options are preserved.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn clear(&mut self) {
    self.buffer.clear();
  }
}

// ---- Helpers ---------------------------------------------------------------

/// `N = clamp(ceil(duration / target_interval), 1, max_frames_per_shot)`
///
/// Uses integer (nanosecond) arithmetic to avoid pulling in `f64::ceil`,
/// which isn't available in `no_std` builds.
fn compute_n_buckets(duration: Duration, opts: &Options) -> usize {
  let target_ns = opts.target_interval.as_nanos();
  if target_ns == 0 {
    return 1;
  }
  // Ceiling division in u128.
  let d_ns = duration.as_nanos();
  let raw = (d_ns.saturating_add(target_ns - 1)) / target_ns;
  let capped = raw.min(opts.max_frames_per_shot as u128).max(1);
  capped as usize
}

/// Returns the `(effective_start, effective_end)` timestamp of each
/// bucket, margin applied to the first and last. When the margin eats
/// the whole bucket (degenerate configuration), falls back to the
/// un-shrunk bucket so the bucket still has a chance to contribute.
fn compute_bucket_ranges(
  range: &TimeRange,
  n: usize,
  margin_ratio: f64,
) -> Vec<(Timestamp, Timestamp)> {
  let mut out = Vec::with_capacity(n);
  let n_f = n as f64;
  for b in 0..n {
    let t0 = (b as f64) / n_f;
    let t1 = ((b + 1) as f64) / n_f;
    let eff_t0 = if b == 0 { t0 + margin_ratio } else { t0 };
    let eff_t1 = if b == n - 1 { t1 - margin_ratio } else { t1 };
    let (use_t0, use_t1) = if eff_t1 > eff_t0 {
      (eff_t0, eff_t1)
    } else {
      (t0, t1) // un-shrunk fallback
    };
    out.push((range.interpolate(use_t0), range.interpolate(use_t1)));
  }
  out
}

/// `true` when `a > b` under `f32::partial_cmp`. NaN compares as not-
/// greater, so a NaN score cannot unseat a numeric incumbent — our
/// reductions do not produce NaN anyway, but the defensive default
/// keeps the running-argmax state well-defined under any input.
#[cfg_attr(not(tarpaulin), inline(always))]
fn sharper(a: f32, b: f32) -> bool {
  a.partial_cmp(&b).unwrap_or(Ordering::Equal) == Ordering::Greater
}

/// Any-one-trips hard gate matching the Python reference's flat /
/// over-/under-exposed / clipped checks.
fn hard_gate(s: &FrameScore, opts: &Options) -> bool {
  if s.brightness < opts.black_mean_threshold as f32 {
    return true;
  }
  if s.brightness > opts.bright_mean_threshold as f32 {
    return true;
  }
  // AND-gate: only flag flat when BOTH variances are low (keeps
  // equiluminant multi-colour frames).
  if s.luma_variance < opts.luma_variance_threshold
    && s.saturation_variance < opts.sat_variance_threshold
  {
    return true;
  }
  if s.clipping > opts.max_clipping {
    return true;
  }
  false
}

// ----------------------------------------------------------------------------

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::frame::{Timebase, Timestamp};
  use core::num::NonZeroU32;

  fn nz(n: u32) -> NonZeroU32 {
    NonZeroU32::new(n).expect("non-zero")
  }

  fn ts(pts: i64) -> Timestamp {
    // 1 µs per tick — makes pts directly readable as microseconds.
    Timestamp::new(pts, Timebase::new(1, nz(1_000_000)))
  }

  fn tr(start_us: i64, end_us: i64) -> TimeRange {
    TimeRange::new(start_us, end_us, Timebase::new(1, nz(1_000_000)))
  }

  fn good_score(sharpness: f32) -> FrameScore {
    FrameScore {
      sharpness,
      brightness: 128.0,
      luma_variance: 200.0,
      saturation_variance: 100.0,
      clipping: 0.0,
    }
  }

  // ----- Options -------------------------------------------------------------

  #[test]
  fn default_options_match_design_doc() {
    let o = Options::default();
    assert_eq!(o.target_interval(), Duration::from_secs(4));
    assert_eq!(o.max_frames_per_shot(), 16);
    assert!((o.margin_ratio() - 0.02).abs() < 1e-9);
    assert_eq!(o.min_sharpness(), 100.0);
    assert_eq!(o.black_mean_threshold(), 15);
    assert_eq!(o.bright_mean_threshold(), 240);
    assert_eq!(o.luma_variance_threshold(), 5.0);
    assert_eq!(o.sat_variance_threshold(), 3.0);
    assert_eq!(o.max_clipping(), 0.5);
  }

  #[test]
  fn options_builders_roundtrip() {
    let o = Options::new()
      .with_target_interval(Duration::from_secs(2))
      .with_max_frames_per_shot(8)
      .with_margin_ratio(0.05)
      .with_min_sharpness(50.0)
      .with_black_mean_threshold(10)
      .with_bright_mean_threshold(245)
      .with_luma_variance_threshold(1.0)
      .with_sat_variance_threshold(2.0)
      .with_max_clipping(0.25);
    assert_eq!(o.target_interval(), Duration::from_secs(2));
    assert_eq!(o.max_frames_per_shot(), 8);
    assert_eq!(o.margin_ratio(), 0.05);
    assert_eq!(o.min_sharpness(), 50.0);
    assert_eq!(o.black_mean_threshold(), 10);
    assert_eq!(o.bright_mean_threshold(), 245);
    assert_eq!(o.luma_variance_threshold(), 1.0);
    assert_eq!(o.sat_variance_threshold(), 2.0);
    assert_eq!(o.max_clipping(), 0.25);
  }

  #[test]
  #[should_panic(expected = "max_frames_per_shot")]
  fn options_max_frames_zero_panics() {
    let _ = Options::new().with_max_frames_per_shot(0);
  }

  #[test]
  #[should_panic(expected = "margin_ratio")]
  fn options_margin_half_panics() {
    let _ = Options::new().with_margin_ratio(0.5);
  }

  #[test]
  #[should_panic(expected = "max_clipping")]
  fn options_max_clipping_out_of_range_panics() {
    let _ = Options::new().with_max_clipping(1.5);
  }

  // ----- compute_n_buckets ---------------------------------------------------

  #[test]
  fn n_buckets_scales_with_duration() {
    let o = Options::default();
    assert_eq!(compute_n_buckets(Duration::from_secs(1), &o), 1);
    assert_eq!(compute_n_buckets(Duration::from_secs(4), &o), 1);
    assert_eq!(compute_n_buckets(Duration::from_secs(5), &o), 2);
    assert_eq!(compute_n_buckets(Duration::from_secs(12), &o), 3);
    assert_eq!(compute_n_buckets(Duration::from_secs(16), &o), 4);
  }

  #[test]
  fn n_buckets_capped_by_max_frames() {
    let o = Options::default().with_max_frames_per_shot(5);
    // 100s with target 4s would ask for 25; clamped to 5.
    assert_eq!(compute_n_buckets(Duration::from_secs(100), &o), 5);
  }

  #[test]
  fn n_buckets_floor_is_one() {
    let o = Options::default();
    assert_eq!(compute_n_buckets(Duration::from_nanos(1), &o), 1);
  }

  // ----- hard_gate -----------------------------------------------------------

  #[test]
  fn hard_gate_rejects_too_dark() {
    let o = Options::default();
    let mut s = good_score(200.0);
    s.brightness = 5.0;
    assert!(hard_gate(&s, &o));
  }

  #[test]
  fn hard_gate_rejects_too_bright() {
    let o = Options::default();
    let mut s = good_score(200.0);
    s.brightness = 250.0;
    assert!(hard_gate(&s, &o));
  }

  #[test]
  fn hard_gate_rejects_flat_frame() {
    let o = Options::default();
    let mut s = good_score(200.0);
    s.luma_variance = 1.0;
    s.saturation_variance = 1.0;
    assert!(hard_gate(&s, &o));
  }

  #[test]
  fn hard_gate_keeps_equiluminant_multicolour() {
    // Low luma variance but high saturation variance — the AND-gate
    // keeps this frame alive.
    let o = Options::default();
    let mut s = good_score(200.0);
    s.luma_variance = 1.0;
    s.saturation_variance = 80.0;
    assert!(!hard_gate(&s, &o));
  }

  #[test]
  fn hard_gate_rejects_heavy_clipping() {
    let o = Options::default();
    let mut s = good_score(200.0);
    s.clipping = 0.9;
    assert!(hard_gate(&s, &o));
  }

  // ----- Detector ------------------------------------------------------------

  #[test]
  fn observe_and_buffered() {
    let mut det = Detector::new(Options::default());
    det.observe(ts(0), good_score(100.0));
    det.observe(ts(1_000), good_score(200.0));
    assert_eq!(det.buffered(), 2);
  }

  #[test]
  fn clear_empties_buffer() {
    let mut det = Detector::new(Options::default());
    det.observe(ts(0), good_score(100.0));
    det.clear();
    assert_eq!(det.buffered(), 0);
  }

  #[test]
  fn finalize_single_bucket_picks_sharpest() {
    // 2-second shot with target_interval=4s → 1 bucket.
    let opts = Options::default().with_margin_ratio(0.0); // disable margin
    let mut det = Detector::new(opts);
    det.observe(ts(0), good_score(100.0));
    det.observe(ts(500_000), good_score(500.0)); // sharpest
    det.observe(ts(1_500_000), good_score(200.0));

    let out = det.finalize_shot(tr(0, 2_000_000));
    assert_eq!(out, vec![ts(500_000)]);
    assert_eq!(det.buffered(), 0, "in-range entries should be drained");
  }

  #[test]
  fn finalize_multiple_buckets_pick_per_bucket_sharpest() {
    // 12s shot with target 4s → 3 buckets; disable margin for clean bounds.
    let opts = Options::default().with_margin_ratio(0.0);
    let mut det = Detector::new(opts);

    // Bucket 0: [0, 4s). Best at 1s.
    det.observe(ts(500_000), good_score(100.0));
    det.observe(ts(1_000_000), good_score(300.0));
    det.observe(ts(3_500_000), good_score(150.0));
    // Bucket 1: [4s, 8s). Best at 5s.
    det.observe(ts(4_500_000), good_score(200.0));
    det.observe(ts(5_000_000), good_score(500.0));
    det.observe(ts(7_500_000), good_score(100.0));
    // Bucket 2: [8s, 12s). Best at 10s.
    det.observe(ts(9_000_000), good_score(150.0));
    det.observe(ts(10_000_000), good_score(450.0));
    det.observe(ts(11_500_000), good_score(200.0));

    let out = det.finalize_shot(tr(0, 12_000_000));
    assert_eq!(out, vec![ts(1_000_000), ts(5_000_000), ts(10_000_000)]);
  }

  #[test]
  fn finalize_falls_back_when_all_frames_fail_gates() {
    // Entire bucket's frames are "bad" (too dark). Fallback picks the
    // sharpest anyway.
    let opts = Options::default().with_margin_ratio(0.0);
    let mut det = Detector::new(opts);
    let bad = |sharp| FrameScore {
      sharpness: sharp,
      brightness: 5.0, // below black threshold 15
      luma_variance: 200.0,
      saturation_variance: 100.0,
      clipping: 0.0,
    };
    det.observe(ts(0), bad(100.0));
    det.observe(ts(1_000_000), bad(400.0)); // sharpest among bad
    det.observe(ts(3_000_000), bad(200.0));

    let out = det.finalize_shot(tr(0, 4_000_000));
    assert_eq!(out, vec![ts(1_000_000)]);
  }

  #[test]
  fn finalize_skips_bucket_with_no_entries() {
    // 3-bucket shot; middle bucket has no observations.
    let opts = Options::default().with_margin_ratio(0.0);
    let mut det = Detector::new(opts);
    det.observe(ts(1_000_000), good_score(300.0));
    // 4..8 s: nothing
    det.observe(ts(9_000_000), good_score(400.0));

    let out = det.finalize_shot(tr(0, 12_000_000));
    assert_eq!(out, vec![ts(1_000_000), ts(9_000_000)]);
  }

  #[test]
  fn finalize_drops_stale_entries_from_earlier_shots() {
    let opts = Options::default().with_margin_ratio(0.0);
    let mut det = Detector::new(opts);
    det.observe(ts(100), good_score(500.0)); // pre-shot, should be dropped
    det.observe(ts(500_000), good_score(200.0));

    let out = det.finalize_shot(tr(200_000, 2_000_000));
    assert_eq!(out, vec![ts(500_000)]);
    assert_eq!(det.buffered(), 0);
  }

  #[test]
  fn finalize_retains_post_shot_entries_for_next_call() {
    let opts = Options::default().with_margin_ratio(0.0);
    let mut det = Detector::new(opts);
    det.observe(ts(500_000), good_score(100.0));
    det.observe(ts(5_000_000), good_score(900.0)); // belongs to next shot

    let out = det.finalize_shot(tr(0, 2_000_000));
    assert_eq!(out, vec![ts(500_000)]);
    assert_eq!(det.buffered(), 1, "future entries preserved");
    let out2 = det.finalize_shot(tr(2_000_000, 6_000_000));
    assert_eq!(out2, vec![ts(5_000_000)]);
  }

  #[test]
  fn finalize_degenerate_range_returns_empty_and_drops_stale() {
    let mut det = Detector::new(Options::default());
    det.observe(ts(100), good_score(100.0));
    det.observe(ts(500_000), good_score(200.0));

    // end == start → zero duration, no emits. Stale (pts < 200_000)
    // entries still dropped.
    let out = det.finalize_shot(tr(200_000, 200_000));
    assert!(out.is_empty());
    assert_eq!(det.buffered(), 1, "only the future entry remains");
  }

  #[test]
  fn finalize_respects_first_bucket_margin() {
    // 10-s shot, 1 bucket. Margin 0.1 → effective bucket is [1s, 9s).
    // An entry at 500 ms should be in the pre-margin gap (skipped),
    // one at 5 s used.
    let opts = Options::default()
      .with_target_interval(Duration::from_secs(20)) // force 1 bucket
      .with_margin_ratio(0.1);
    let mut det = Detector::new(opts);
    det.observe(ts(500_000), good_score(900.0)); // pre-margin
    det.observe(ts(5_000_000), good_score(300.0)); // in-bucket
    det.observe(ts(9_500_000), good_score(800.0)); // post-margin

    let out = det.finalize_shot(tr(0, 10_000_000));
    assert_eq!(out, vec![ts(5_000_000)]);
  }

  #[test]
  fn finalize_emits_in_pts_order() {
    let opts = Options::default().with_margin_ratio(0.0);
    let mut det = Detector::new(opts);
    det.observe(ts(1_000_000), good_score(100.0));
    det.observe(ts(5_000_000), good_score(100.0));
    det.observe(ts(9_000_000), good_score(100.0));
    let out = det.finalize_shot(tr(0, 12_000_000));
    assert!(out.windows(2).all(|w| w[0].pts() < w[1].pts()));
  }

  #[test]
  fn finalize_can_be_called_multiple_times() {
    let opts = Options::default().with_margin_ratio(0.0);
    let mut det = Detector::new(opts);
    det.observe(ts(500_000), good_score(100.0));
    det.observe(ts(5_000_000), good_score(100.0));
    let out1 = det.finalize_shot(tr(0, 2_000_000));
    let out2 = det.finalize_shot(tr(2_000_000, 6_000_000));
    assert_eq!(out1.len(), 1);
    assert_eq!(out2.len(), 1);
  }

  #[test]
  fn finalize_remaining_treats_buffer_tail_as_final_shot() {
    let opts = Options::default().with_margin_ratio(0.0);
    let mut det = Detector::new(opts);
    // First shot closed normally.
    det.observe(ts(500_000), good_score(100.0));
    let _ = det.finalize_shot(tr(0, 2_000_000));
    // Second shot opens but EOS arrives before a confirmed cut.
    det.observe(ts(3_000_000), good_score(200.0));
    det.observe(ts(4_500_000), good_score(400.0));

    let out = det.finalize_remaining(ts(6_000_000));
    assert_eq!(out, vec![ts(4_500_000)]);
    assert_eq!(det.buffered(), 0);
  }

  #[test]
  fn finalize_remaining_empty_buffer_returns_empty() {
    let mut det = Detector::new(Options::default());
    let out = det.finalize_remaining(ts(1_000_000));
    assert!(out.is_empty());
  }

  #[test]
  fn finalize_remaining_eos_before_buffer_start_returns_empty() {
    let mut det = Detector::new(Options::default());
    det.observe(ts(5_000_000), good_score(100.0));
    let out = det.finalize_remaining(ts(1_000_000));
    assert!(out.is_empty(), "eos before earliest buffered ts should no-op");
  }

  #[test]
  #[should_panic(expected = "target_interval must be > 0")]
  fn options_target_interval_zero_panics() {
    let _ = Options::new().with_target_interval(Duration::ZERO);
  }

  #[test]
  fn sharper_returns_ordering_greater() {
    assert!(sharper(2.0, 1.0));
    assert!(!sharper(1.0, 2.0));
    assert!(!sharper(1.0, 1.0));
    // NaN tolerant — not-greater either way.
    assert!(!sharper(f32::NAN, 1.0));
    assert!(!sharper(1.0, f32::NAN));
  }
}
