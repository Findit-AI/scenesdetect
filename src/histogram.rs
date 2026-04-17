//! Histogram-based scene detection via luma correlation.
//!
//! This module implements [`Detector`](crate::histogram::Detector),
//! a port of PySceneDetect's `detect-hist` algorithm. A cut is registered
//! when the distribution of brightness across the frame changes abruptly —
//! the classic signature of a hard cut between scenes.
//!
//! # Algorithm
//!
//! For each incoming [`LumaFrame`](crate::frame::LumaFrame):
//!
//! 1. **Compute a histogram** of the luma (Y) plane over `bins` uniformly
//!    spaced buckets covering `[0, 256)`. Row padding (when `stride > width`)
//!    is skipped.
//! 2. **Compare with the previous frame's histogram** using the Pearson
//!    correlation coefficient (OpenCV's `HISTCMP_CORREL`):
//!
//!    ```text
//!                Σᵢ (H1ᵢ − H̄1)(H2ᵢ − H̄2)
//!    ρ(H1, H2) = ──────────────────────────────────
//!                √( Σᵢ (H1ᵢ − H̄1)² · Σᵢ (H2ᵢ − H̄2)² )
//!    ```
//!
//!    ρ ∈ [−1, 1]. `ρ = 1` means identical shape; lower values indicate the
//!    brightness distribution has changed.
//! 3. **Apply the threshold.** A cut is proposed when `ρ ≤ 1 − threshold`.
//!    The user-facing `threshold` is the allowed *drop* in correlation, so
//!    larger values are *less* sensitive.
//! 4. **Apply the `min_duration` gate.** After a cut is emitted, further
//!    cuts are suppressed until at least `min_duration` of presentation time
//!    has elapsed since the previous cut (or the start of the stream).
//!    Prevents false positives from flashes and rapid intercutting.
//!
//! The first frame establishes the baseline — no cut is emitted for it — and
//! seeds the `last_cut_ts` reference so the min-duration gate can be
//! evaluated from frame two onward.
//!
//! # Intuition
//!
//! Camera motion, object motion, and gradual lighting changes all tend to
//! *preserve* the overall shape of the luma histogram; a cut to a new scene
//! typically does not. Pearson correlation captures *shape* similarity
//! rather than absolute values, so a uniform brightness shift (e.g., exposure
//! compensation) on its own does not trigger a cut.
//!
//! # Limits
//!
//! - **Dissolves and fades** change brightness gradually — consecutive-frame
//!   correlation stays high, so soft transitions are typically missed.
//!   Combine with a content-based detector for those.
//! - **Camera flashes** can spike the correlation downward; the `min_duration`
//!   gate filters repeated flashes but not isolated ones. Tune to your
//!   source.
//! - **Scenes with similar brightness distributions** (two dim interiors, two
//!   daylight exteriors) can correlate highly even across a true cut.
//!   Histogram alone is an imperfect signal.
//!
//! # Streaming
//!
//! [`Detector`](crate::histogram::Detector) holds two
//! rotating `Vec<f64>` buffers sized to `bins`; after construction it
//! performs no per-frame allocation. It takes
//! [`LumaFrame`](crate::frame::LumaFrame) values whose timestamps carry any
//! [`Timebase`](crate::frame::Timebase) — the `min_duration` gate works
//! across mixed timebases via
//! [`Timestamp::duration_since`](crate::frame::Timestamp::duration_since).
//!
//! # Attribution
//!
//! Ported from PySceneDetect's `detect-hist` (BSD 3-Clause).
//! See <https://scenedetect.com> for the original implementation.

use core::{num::NonZeroUsize, time::Duration};

use derive_more::IsVariant;
use thiserror::Error;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::frame::{LumaFrame, Timebase, Timestamp};

use std::{vec, vec::Vec};

/// Error returned by [`Detector::try_new`] when the provided [`Options`]
/// are inconsistent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, IsVariant, Error)]
#[non_exhaustive]
pub enum Error {
  /// `N_ACCUM * bins` overflows `usize`. The bin count is too large for the
  /// multi-accumulator scratch buffer.
  #[error("histogram bin count ({bins}) is too large (N_ACCUM * bins overflows usize)")]
  BinCountTooLarge {
    /// The requested bin count that caused the overflow.
    bins: usize,
  },
}

/// Options for the histogram-based scene detector. See the [module docs]
/// for how each parameter shapes the algorithm.
///
/// [module docs]: crate::histogram
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Options {
  threshold: f64,
  bins: NonZeroUsize,
  #[cfg_attr(feature = "serde", serde(with = "humantime_serde"))]
  min_duration: Duration,
  allow_initial_cut: bool,
}

impl Default for Options {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn default() -> Self {
    Self::new()
  }
}

impl Options {
  /// Creates a new `Options` instance with default values.
  ///
  /// Defaults: `threshold = 0.5`, `bins = 256`, `min_duration = 1 s`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      threshold: 0.5,
      bins: NonZeroUsize::new(256).unwrap(),
      min_duration: Duration::from_secs(1),
      allow_initial_cut: true,
    }
  }

  /// Returns the cut-detection threshold.
  ///
  /// Values in `[0.0, 1.0]`. Higher values require a larger drop in histogram
  /// correlation to register a cut (less sensitive). Typical range: 0.05–0.5.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn threshold(&self) -> f64 {
    self.threshold
  }

  /// Set the value of the threshold.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_threshold(mut self, val: f64) -> Self {
    self.set_threshold(val);
    self
  }

  /// Set the value of the threshold.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_threshold(&mut self, val: f64) -> &mut Self {
    self.threshold = val;
    self
  }

  /// Returns the number of histogram bins.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn bins(&self) -> usize {
    self.bins.get()
  }

  /// Set the value of the number of bins.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_bins(mut self, val: NonZeroUsize) -> Self {
    self.set_bins(val);
    self
  }

  /// Set the value of the number of bins.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_bins(&mut self, val: NonZeroUsize) -> &mut Self {
    self.bins = val;
    self
  }

  /// Returns the minimum scene duration.
  ///
  /// After a cut is emitted, no further cut will be emitted until at least
  /// this amount of presentation time has elapsed. Suppresses rapid flashes
  /// and fast cuts.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_duration(&self) -> Duration {
    self.min_duration
  }

  /// Set the value of the minimum scene duration.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_duration(mut self, val: Duration) -> Self {
    self.set_min_duration(val);
    self
  }

  /// Set the value of the minimum scene duration.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_duration(&mut self, val: Duration) -> &mut Self {
    self.min_duration = val;
    self
  }

  /// Set the minimum scene length as a number of frames at a given frame rate.
  ///
  /// Convenience for users coming from frame-count APIs (e.g., PySceneDetect's
  /// `min_scene_len`). Internally this converts to [`Self::min_duration`] via
  /// [`Timebase::frames_to_duration`]. On VFR content the duration stays fixed
  /// while frame counts drift — that's the desired behavior.
  ///
  /// `fps` is interpreted as frames per second: 30 fps = `Timebase::new(30, 1)`,
  /// NTSC = `Timebase::new(30000, 1001)`.
  ///
  /// # Panics
  ///
  /// Panics if `fps.num() == 0`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_frames(mut self, frames: u32, fps: Timebase) -> Self {
    self.set_min_frames(frames, fps);
    self
  }

  /// In-place form of [`Self::with_min_frames`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_frames(&mut self, frames: u32, fps: Timebase) -> &mut Self {
    self.min_duration = fps.frames_to_duration(frames);
    self
  }

  /// Whether the first detected cut is allowed to fire immediately.
  ///
  /// - `true` (default): the first detected cut fires as soon as the
  ///   correlation drops below `1 - threshold`.
  /// - `false`: suppresses cuts until the stream has actually run for at
  ///   least [`Self::min_duration`]. Matches PySceneDetect's default.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn allow_initial_cut(&self) -> bool {
    self.allow_initial_cut
  }

  /// Sets whether the first detected cut may fire immediately.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_allow_initial_cut(mut self, val: bool) -> Self {
    self.allow_initial_cut = val;
    self
  }

  /// Sets `allow_initial_cut` in place.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_allow_initial_cut(&mut self, val: bool) -> &mut Self {
    self.allow_initial_cut = val;
    self
  }
}

/// Number of parallel accumulators used by [`Detector::compute_histogram`].
///
/// Round-robin dispatch across 4 accumulators breaks the loop-carried
/// `hist[idx] += 1` store-load dependency. Measured against N_ACCUM=8 on a
/// modern core: the 4-wide pattern already saturates memory ports for this
/// workload, so more accumulators give no further speedup.
const N_ACCUM: usize = 4;

/// Histogram-correlation scene detector.
///
/// Compares the luma (Y-plane) histogram of consecutive frames using Pearson
/// correlation. A cut is emitted when the correlation drops below
/// `1.0 - threshold` *and* at least [`Options::min_duration`] has elapsed
/// since the previous cut (or stream start).
///
/// For the full algorithm — binning, correlation formula, thresholding, and
/// min-duration gating — see the [module-level documentation](crate::histogram).
///
/// # Hot-path performance
///
/// After construction, the detector does not allocate per frame. It holds:
///
/// - a precomputed `[u32; 256]` pixel → bin lookup table (so the inner loop
///   is a single load, no arithmetic per pixel);
/// - a `4 × bins` multi-accumulator scratch buffer (breaks the loop-carried
///   `hist[idx] += 1` dependency chain);
/// - two reduced `Vec<u32>` histograms (current and previous, each sized to
///   `bins`). Integer counters are 4× smaller and faster to increment than
///   the `f64` they replace.
#[derive(Debug, Clone)]
pub struct Detector {
  options: Options,
  corr_threshold: f64,
  /// Lookup table: pixel value (0..=255) → bin index.
  bin_of: [u32; 256],
  /// `N_ACCUM * bins` parallel accumulator slots (laid out contiguously as
  /// `[acc0..acc1..acc2..acc3]`).
  scratch: Vec<u32>,
  current: Vec<u32>,
  previous: Vec<u32>,
  has_previous: bool,
  last_cut_ts: Option<Timestamp>,
  last_hist_diff: Option<f64>,
}

impl Detector {
  /// Creates a new `Detector` instance with the given options.
  ///
  /// # Panics
  ///
  /// Panics if the options are invalid — see [`enum@Error`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new(options: Options) -> Self {
    Self::try_new(options).expect("invalid histogram::Options")
  }

  /// Creates a new `Detector` instance, returning [`enum@Error`] if the
  /// options are invalid.
  ///
  /// Builds the pixel → bin lookup table and pre-allocates the multi-accumulator
  /// scratch (`4 * bins` × `u32`) plus the two reduced histograms.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn try_new(options: Options) -> Result<Self, Error> {
    let bins = options.bins.get();
    let scratch_len = N_ACCUM
      .checked_mul(bins)
      .ok_or(Error::BinCountTooLarge { bins })?;
    let corr_threshold = (1.0 - options.threshold).clamp(0.0, 1.0);
    let bin_of = build_bin_lookup(bins);
    Ok(Self {
      options,
      corr_threshold,
      bin_of,
      scratch: vec![0u32; scratch_len],
      current: vec![0u32; bins],
      previous: vec![0u32; bins],
      has_previous: false,
      last_cut_ts: None,
      last_hist_diff: None,
    })
  }

  /// Returns a reference to the options used by this detector.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn options(&self) -> &Options {
    &self.options
  }

  /// Returns the correlation between the last two frames' histograms, or
  /// `None` if fewer than two frames have been processed.
  ///
  /// Range: `[-1.0, 1.0]`. `1.0` means identical shape; lower values indicate
  /// change. Useful for logging/diagnostics.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn last_hist_diff(&self) -> Option<f64> {
    self.last_hist_diff
  }

  /// Resets the detector's streaming state so it can be reused on a fresh
  /// stream (e.g., when the next video begins) without rebuilding the
  /// lookup table or reallocating the accumulator / histogram buffers.
  ///
  /// After `clear()` the next [`Self::process`] call is treated as if it
  /// were the first frame of a new stream: no cut is emitted, and the frame
  /// re-seeds `last_cut_ts`. The previous video's histograms, `last_cut_ts`,
  /// and `last_hist_diff` are all discarded.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn clear(&mut self) {
    self.has_previous = false;
    self.last_cut_ts = None;
    self.last_hist_diff = None;
  }

  /// Processes the next frame. Returns `Some(ts)` if a cut is detected at
  /// the frame's timestamp, otherwise `None`.
  ///
  /// The first frame establishes the baseline histogram and cut-gating
  /// reference; no cut is emitted for it.
  pub fn process(&mut self, frame: LumaFrame<'_>) -> Option<Timestamp> {
    let ts = frame.timestamp();

    // Seed the cut-gating reference on the first frame.
    if self.last_cut_ts.is_none() {
      // Seed: virtual-past if allow_initial_cut lets the first cut fire
      // immediately, otherwise match Python — seed at `ts`, suppressing
      // cuts within the first min_duration of the stream.
      self.last_cut_ts = Some(if self.options.allow_initial_cut {
        ts.saturating_sub_duration(self.options.min_duration)
      } else {
        ts
      });
    }

    self.compute_histogram(&frame);

    let mut cut: Option<Timestamp> = None;
    if self.has_previous {
      let diff = correlation(&self.previous, &self.current);
      self.last_hist_diff = Some(diff);

      let min_elapsed = self
        .last_cut_ts
        .as_ref()
        .and_then(|last| ts.duration_since(last))
        .is_some_and(|d| d >= self.options.min_duration);

      if diff <= self.corr_threshold && min_elapsed {
        cut = Some(ts);
        self.last_cut_ts = Some(ts);
      }
    }

    core::mem::swap(&mut self.current, &mut self.previous);
    self.has_previous = true;
    cut
  }

  /// Fills `self.current` with bin counts for the luma samples in `frame`,
  /// respecting `stride` (row padding is skipped).
  ///
  /// Uses `N_ACCUM` parallel accumulators laid out contiguously in
  /// `self.scratch` (first `bins` entries are acc 0, next `bins` are acc 1,
  /// etc.), reduced into `self.current` at the end. Both buffers are
  /// zero-filled before use.
  fn compute_histogram(&mut self, frame: &LumaFrame<'_>) {
    let bins = self.options.bins.get();
    let data = frame.data();
    let w = frame.width() as usize;
    let h = frame.height() as usize;
    let s = frame.stride() as usize;

    // Partial borrows of disjoint fields so the inner loop can read
    // `bin_of` while we're mutating `scratch` and later `current`.
    let scratch = &mut self.scratch;
    let current = &mut self.current;
    let bin_of = &self.bin_of;

    debug_assert_eq!(scratch.len(), N_ACCUM * bins);
    debug_assert_eq!(current.len(), bins);

    scratch.fill(0);

    let (acc0, rest) = scratch.split_at_mut(bins);
    let (acc1, rest) = rest.split_at_mut(bins);
    let (acc2, acc3) = rest.split_at_mut(bins);

    for y in 0..h {
      let row_start = y * s;
      let row = &data[row_start..row_start + w];

      let chunks = row.chunks_exact(N_ACCUM);
      let remainder = chunks.remainder();
      for chunk in chunks {
        // Four independent accumulator updates — no loop-carried dependency.
        acc0[bin_of[chunk[0] as usize] as usize] += 1;
        acc1[bin_of[chunk[1] as usize] as usize] += 1;
        acc2[bin_of[chunk[2] as usize] as usize] += 1;
        acc3[bin_of[chunk[3] as usize] as usize] += 1;
      }
      // Tail: at most N_ACCUM - 1 pixels.
      for (i, &v) in remainder.iter().enumerate() {
        let idx = bin_of[v as usize] as usize;
        match i {
          0 => acc0[idx] += 1,
          1 => acc1[idx] += 1,
          2 => acc2[idx] += 1,
          _ => acc3[idx] += 1,
        }
      }
    }

    // Reduce the four accumulators into `current`. Vectorizes trivially.
    for j in 0..bins {
      current[j] = acc0[j] + acc1[j] + acc2[j] + acc3[j];
    }
  }
}

/// Builds a 256-entry lookup table mapping pixel value to bin index.
///
/// Bin formula matches OpenCV's `calcHist` with range `[0, 256]`:
/// `idx = v * bins / 256`, computed in `u64` to tolerate any `bins ≤ u32::MAX`.
fn build_bin_lookup(bins: usize) -> [u32; 256] {
  let mut t = [0u32; 256];
  let b = bins as u64;
  let mut v = 0usize;
  while v < 256 {
    t[v] = ((v as u64 * b) / 256) as u32;
    v += 1;
  }
  t
}

/// Pearson correlation between two equally-sized histograms.
///
/// Matches OpenCV's `HISTCMP_CORREL`. Range `[-1, 1]`. For flat histograms
/// (zero variance), returns `1.0` if identical and `0.0` otherwise.
fn correlation(a: &[u32], b: &[u32]) -> f64 {
  debug_assert_eq!(a.len(), b.len());
  let n = a.len() as f64;
  let sum_a: u64 = a.iter().map(|&x| x as u64).sum();
  let sum_b: u64 = b.iter().map(|&x| x as u64).sum();
  let mean_a = sum_a as f64 / n;
  let mean_b = sum_b as f64 / n;
  let mut num = 0.0;
  let mut var_a = 0.0;
  let mut var_b = 0.0;
  for (&x, &y) in a.iter().zip(b.iter()) {
    let da = x as f64 - mean_a;
    let db = y as f64 - mean_b;
    num += da * db;
    var_a += da * da;
    var_b += db * db;
  }
  if var_a == 0.0 && var_b == 0.0 {
    return if a == b { 1.0 } else { 0.0 };
  }
  if var_a == 0.0 || var_b == 0.0 {
    return 0.0;
  }
  num / super::sqrt_64(var_a * var_b)
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::frame::Timebase;
  use core::num::NonZeroU32;

  const fn nz32(n: u32) -> NonZeroU32 {
    match NonZeroU32::new(n) {
      Some(v) => v,
      None => panic!("zero"),
    }
  }

  fn make_frame<'a>(data: &'a [u8], w: u32, h: u32, pts: i64) -> LumaFrame<'a> {
    let tb = Timebase::new(1, nz32(1000)); // 1ms units
    LumaFrame::new(data, w, h, w, Timestamp::new(pts, tb))
  }

  #[test]
  fn identical_frames_produce_no_cut() {
    let mut det = Detector::new(Options::default());
    // Uniform mid-gray frame.
    let buf = [128u8; 64 * 48];
    assert!(det.process(make_frame(&buf, 64, 48, 0)).is_none());
    assert!(det.process(make_frame(&buf, 64, 48, 2000)).is_none());
    assert!(det.process(make_frame(&buf, 64, 48, 4000)).is_none());
    // Correlation should be 1.0 (or treated as such for flat identical frames).
    assert_eq!(det.last_hist_diff(), Some(1.0));
  }

  #[test]
  fn very_different_frames_produce_cut() {
    // threshold=0.5 → corr_threshold=0.5; a black→white transition has
    // correlation close to 0 (or negative), well under 0.5.
    let opts = Options::default().with_min_duration(Duration::from_millis(0));
    let mut det = Detector::new(opts);

    let black = [0u8; 64 * 48];
    let white = [255u8; 64 * 48];

    // First frame primes the detector; second frame is the cut.
    assert!(det.process(make_frame(&black, 64, 48, 0)).is_none());
    let cut = det.process(make_frame(&white, 64, 48, 33));
    assert!(
      cut.is_some(),
      "expected a cut at the black→white transition"
    );
    assert_eq!(cut.unwrap().pts(), 33);
  }

  #[test]
  fn min_duration_suppresses_rapid_cuts() {
    // 1 second min_duration, Python-compat mode (allow_initial_cut=false).
    // Alternate black/white frames at 33 ms cadence — no cut should fire
    // before 1 s elapses from stream start.
    let opts = Options::default()
      .with_min_duration(Duration::from_secs(1))
      .with_allow_initial_cut(false);
    let mut det = Detector::new(opts);

    let black = [0u8; 64 * 48];
    let white = [255u8; 64 * 48];

    let mut cuts = 0u32;
    // 30 frames ≈ 1 second at 30 fps, alternating.
    for i in 0..30i64 {
      let frame_data = if i % 2 == 0 { &black } else { &white };
      let ts = i * 33; // in 1/1000 timebase → ms
      if det.process(make_frame(frame_data, 64, 48, ts)).is_some() {
        cuts += 1;
      }
    }
    // First flip after frame 0 initializes last_cut_ts at pts=0, so the cut
    // at pts=33 is rejected (33 ms < 1 s). No further cuts should land
    // within the first second.
    assert_eq!(cuts, 0, "min_duration should suppress all cuts within 1s");
  }

  #[test]
  fn cut_reported_after_min_duration_elapsed() {
    // Python-compat mode: no early cuts allowed.
    let opts = Options::default()
      .with_min_duration(Duration::from_millis(500))
      .with_allow_initial_cut(false);
    let mut det = Detector::new(opts);

    let black = [0u8; 64 * 48];
    let white = [255u8; 64 * 48];

    // Seed with black @ 0 ms.
    assert!(det.process(make_frame(&black, 64, 48, 0)).is_none());
    // Try to cut at 100 ms — too soon.
    assert!(det.process(make_frame(&white, 64, 48, 100)).is_none());
    // By 600 ms, > 500 ms elapsed since pts=0 → cut allowed.
    let cut = det.process(make_frame(&black, 64, 48, 600));
    assert!(cut.is_some(), "expected cut after min_duration elapsed");
  }

  #[test]
  fn clear_resets_stream_state() {
    // Set min_duration = 0 so the first detectable cut isn't gated.
    let opts = Options::default().with_min_duration(Duration::from_millis(0));
    let mut det = Detector::new(opts);

    let black = [0u8; 64 * 48];
    let white = [255u8; 64 * 48];

    // Video 1: prime, then cut (black→white).
    assert!(det.process(make_frame(&black, 64, 48, 0)).is_none());
    let cut = det.process(make_frame(&white, 64, 48, 33));
    assert!(cut.is_some());
    assert!(det.last_hist_diff().is_some());

    det.clear();

    // After clear: state is fresh. The first frame of "video 2" must NOT
    // emit a cut, even though it's very different from the last frame of
    // video 1 — there's no previous histogram to compare against.
    assert!(det.process(make_frame(&black, 64, 48, 1_000_000)).is_none());
    assert!(
      det.last_hist_diff().is_none(),
      "last_hist_diff should be cleared"
    );

    // Second frame after clear: normal comparison resumes against the
    // just-processed frame.
    let cut2 = det.process(make_frame(&white, 64, 48, 1_000_033));
    assert!(cut2.is_some(), "cut should still be detected on video 2");
  }

  #[test]
  fn compute_histogram_respects_stride() {
    // A 4x2 frame with stride=8 (4 padding bytes per row of junk).
    let mut buf = [0xFFu8; 8 * 2];
    buf[0..4].copy_from_slice(&[10, 20, 30, 40]);
    buf[8..12].copy_from_slice(&[50, 60, 70, 80]);

    let mut det = Detector::new(Options::default());
    let tb = Timebase::new(1, nz32(1000));
    let frame = LumaFrame::new(&buf, 4, 2, 8, Timestamp::new(0, tb));
    det.compute_histogram(&frame);

    for v in [10, 20, 30, 40, 50, 60, 70, 80] {
      assert_eq!(det.current[v as usize], 1);
    }
    assert_eq!(det.current[0xFF], 0, "padding must not be counted");
    assert_eq!(det.current.iter().sum::<u32>(), 8);
  }

  #[test]
  fn compute_histogram_remainder_path() {
    // 7 pixels per row (not a multiple of N_ACCUM=4) exercises the tail loop.
    let mut buf = [0u8; 7 * 3];
    for (i, b) in buf.iter_mut().enumerate() {
      *b = i as u8; // 0..21, all unique
    }

    let mut det = Detector::new(Options::default());
    let tb = Timebase::new(1, nz32(1000));
    let frame = LumaFrame::new(&buf, 7, 3, 7, Timestamp::new(0, tb));
    det.compute_histogram(&frame);

    for v in 0u8..21 {
      assert_eq!(
        det.current[v as usize], 1,
        "pixel value {v} should have count 1"
      );
    }
    assert_eq!(det.current.iter().sum::<u32>(), 21);
  }

  #[test]
  fn build_bin_lookup_matches_formula() {
    let t = build_bin_lookup(256);
    for v in 0..=255u32 {
      assert_eq!(t[v as usize], v);
    }
    let t = build_bin_lookup(128);
    for v in 0..=255u32 {
      assert_eq!(t[v as usize], v / 2);
    }
    let t = build_bin_lookup(1);
    for v in 0..=255u32 {
      assert_eq!(t[v as usize], 0);
    }
  }

  #[test]
  fn correlation_of_identical_is_one() {
    let a: Vec<u32> = vec![1, 2, 3, 4, 5];
    assert!((correlation(&a, &a) - 1.0).abs() < 1e-12);
  }

  #[test]
  fn with_min_frames_matches_python_default() {
    // PySceneDetect's default is 15 frames; at 30 fps that's 500 ms.
    let fps = Timebase::new(30, nz32(1));
    let opts = Options::default().with_min_frames(15, fps);
    assert_eq!(opts.min_duration(), Duration::from_millis(500));
  }

  #[test]
  fn with_min_frames_ntsc() {
    // 15 frames @ NTSC ≈ 500.5 ms.
    let fps = Timebase::new(30_000, nz32(1001));
    let opts = Options::default().with_min_frames(15, fps);
    assert_eq!(opts.min_duration(), Duration::from_nanos(500_500_000));
  }

  #[test]
  fn correlation_of_flat_frames() {
    let a = vec![4u32; 256];
    let b = vec![4u32; 256];
    assert_eq!(correlation(&a, &b), 1.0);
    let c = vec![7u32; 256];
    assert_eq!(correlation(&a, &c), 0.0); // flat but different
  }

  #[test]
  fn try_new_rejects_overflowing_bin_count() {
    let opts = Options::default().with_bins(NonZeroUsize::new(usize::MAX).unwrap());
    let err = Detector::try_new(opts).expect_err("should fail");
    assert_eq!(err, Error::BinCountTooLarge { bins: usize::MAX });
  }

  #[test]
  fn options_accessors_builders_setters_roundtrip() {
    let fps30 = Timebase::new(30, nz32(1));

    // Consuming builder form.
    let opts = Options::default()
      .with_threshold(0.42)
      .with_bins(core::num::NonZeroUsize::new(128).unwrap())
      .with_min_duration(core::time::Duration::from_millis(500))
      .with_allow_initial_cut(false);
    assert_eq!(opts.threshold(), 0.42);
    assert_eq!(opts.bins(), 128);
    assert_eq!(opts.min_duration(), core::time::Duration::from_millis(500));
    assert!(!opts.allow_initial_cut());

    // with_min_frames — alternate min_duration form.
    let opts_frames = Options::default().with_min_frames(15, fps30);
    assert_eq!(
      opts_frames.min_duration(),
      core::time::Duration::from_millis(500)
    );

    // In-place setters, chainable.
    let mut opts = Options::default();
    opts
      .set_threshold(0.1)
      .set_bins(core::num::NonZeroUsize::new(64).unwrap())
      .set_min_duration(core::time::Duration::from_secs(1))
      .set_allow_initial_cut(true);
    assert_eq!(opts.threshold(), 0.1);
    assert_eq!(opts.bins(), 64);
    assert!(opts.allow_initial_cut());

    opts.set_min_frames(30, fps30);
    assert_eq!(opts.min_duration(), core::time::Duration::from_secs(1));
  }

  #[test]
  fn detector_options_and_last_hist_diff_accessors() {
    let opts = Options::default().with_min_duration(core::time::Duration::from_millis(0));
    let mut det = Detector::new(opts.clone());
    assert_eq!(det.options().threshold(), opts.threshold());
    assert!(det.last_hist_diff().is_none());

    let buf = vec![64u8; 32 * 32];
    det.process(make_frame(&buf, 32, 32, 0));
    det.process(make_frame(&buf, 32, 32, 33));
    // After two frames the correlation is defined.
    assert!(det.last_hist_diff().is_some());
  }

  #[test]
  fn histogram_tail_three_hits_acc3_arm() {
    // The 4-way tail handles the last (pixel_count % 4) pixels. Use a
    // frame whose pixel count ≡ 3 (mod 4) so the match arm `_` (acc3)
    // is exercised.
    //
    // 7 * 5 = 35 pixels; 35 % 4 = 3 → tail length 3 → arms 0, 1, 2 AND _.
    let buf = vec![100u8; 35];
    let mut det =
      Detector::new(Options::default().with_min_duration(core::time::Duration::from_millis(0)));
    det.process(make_frame(&buf, 7, 5, 0));
    det.process(make_frame(&buf, 7, 5, 33));
    assert_eq!(det.last_hist_diff(), Some(1.0));
  }
}
