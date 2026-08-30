# Changelog

All notable changes to this crate are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the project adheres to [Semantic Versioning](https://semver.org/).

## 0.4.0

A minor, not a patch: nothing here breaks a build — the new variant
lands on a `#[non_exhaustive]` enum and every signature is unchanged —
but the `cascade` event stream itself is different. Each shot now
carries one keyframe it did not carry before, so any consumer that
counts keyframes, or that treated every keyframe as a quality verdict,
sees new numbers. That is a feature addition with observable fallout,
which is a minor under this crate's 0.x-minor-is-the-boundary policy.

### Added

- **The boundary keyframe.** Opening a shot now emits its first frame
  as an [`Event::Keyframe`](src/cascade/mod.rs) carrying the new
  `Provenance::Boundary(FrameMetrics)` — unconditionally. Being the cut
  is that frame's qualification, so the quality gates never judge it;
  they go on governing only the interval picks that follow. The law
  holds at all three places a shot opens: the stream's first frame, the
  immediate reopen at every confirmed cut, and the reopen a deferred or
  end-of-stream fade cut performs inside `finalize`. It holds for a
  shot whose windows select nothing at all, which is now the ordinary
  shape of a very short shot rather than a shot with no coverage.

  Two properties a consumer may lean on, and which the suite pins:
  every emitted shot's first keyframe sits at exactly that shot's range
  start (so the boundary is index 0 of every shot's keyframe list), and
  no other keyframe ever shares its timestamp.

  The payload is the opening frame's own `FrameMetrics` whenever the
  boundary timestamp names a frame the selector still holds — the
  ordinary case, and free, since those metrics were computed on that
  frame's own push and are read back rather than recomputed (the new
  crate-internal `select::Detector::metrics_at` does the read; nothing
  is rescored). It reads all-zero, as `Provenance::Fallback`'s skipped
  metrics already do, when the boundary names no such frame: an
  interpolated fade cut lands *between* two frames and so names an
  instant no frame occupies.

  End of stream opens no shot and therefore owes no boundary keyframe:
  `finalize` closes the trailing shot one tick past the last frame
  without emitting anything at that instant, which would otherwise
  strand a keyframe outside every emitted range.

### Changed

- **The frame at a cut is no longer selected twice.** It stays buffered
  past its own cut (the finalize range is half-open) and so contends in
  the new shot's first window, where it could win. It has already left
  as that shot's boundary keyframe, so the interval lane now drops the
  winner that matches it. No coverage is lost — the boundary *is* that
  window's representative — but a stream whose window winners were
  already the shot-opening frames emits the same *number* of keyframes
  as before, with the first of each shot re-attributed from
  `Provenance::Quality` / `Provenance::Fallback` to
  `Provenance::Boundary`.
- **Backlog shedding prefers interval picks.** Past
  `MAX_PENDING_OUTPUTS` the oldest queued keyframe is still shed and
  scenes are still never shed, but a boundary keyframe is now shed only
  once no interval pick is left to shed instead. Boundary keyframes
  enter at most once per push, exactly as scenes do, so the backlog
  stays bounded on the same argument while the index-0 property
  survives a slow drain.
- **The `mediaframe` feature's version floor moves to `"0.9"`,** up
  from `"0.8"`. The bump is free: the feature's own RGB-adapter module
  (`frame::mediaframe`) does not exist in this crate — the declaration
  is commented out in `src/frame.rs` — so enabling the feature compiles
  `half` and three `mediaframe/rgb*` sub-features to reach no code of
  this crate's own. No public API moves, because none of this crate's
  public API is gated by the feature.

## 0.3.2

### Changed

- **`mediatime` `0.3` → `0.4`**, additive-only upstream: a new unsigned
  `Duration` counterpart to `SignedDuration`, plus its
  `core::time::Duration` and `SignedDuration` conversions. Upstream's own
  changelog is `### Added` only for 0.4.0 — no `Changed`/`Removed` section
  at all — and states plainly that `Timebase`'s public surface is
  unchanged; `Timestamp` and `TimeRange`, the other two types this crate
  re-exports from [`frame`](src/frame.rs), do not appear in the entry
  either. `cargo check`/`clippy`/`test` across the feature lanes below all
  pass with no source change required.
- **`mediaframe` `0.5` → `0.7`**, for graph coherence only — still a
  patch, not a minor, because nothing of either release is visible from
  here. 0.6's two breaking counts — the `Ch`-prefixed rename of
  `audio::ChannelLayout`'s twelve numeric variants, the human-readable
  `audio::BitRateMode` serde shape — live in `mediaframe`'s `audio`
  module, gated `#[cfg(any(feature = "std", feature = "alloc"))]` there,
  a tier this crate's `mediaframe` feature never reaches (the pin stays
  at the no-alloc tier: `rgb`, `rgb-float`, `rgb-legacy`). 0.7 is
  breaking for a narrower reason — it is `mediaframe` re-bumping its own
  `mediatime` pin `0.3` → `0.4` under the same 0.x-minor-is-the-boundary
  policy as the entry above, not a source change of its own
  (`mediaframe`'s 0.7.0 changelog reports zero fallout in its own source
  either). Either way, the `mediaframe` feature still carries no code
  (`frame::mediaframe` is still commented out) and no `mediaframe::` path
  exists anywhere in this crate, so no `mediaframe` type — and no
  `mediatime` type reached only through `mediaframe` — touches this
  crate's public API.

## 0.3.1

### Changed

- **`mediaframe` `0.4` → `0.5`**, for graph coherence only — a patch,
  not a minor, because nothing of it is visible from here. The
  `mediaframe` feature still carries no code (`frame::mediaframe` is
  still commented out) and no `mediaframe::` path exists anywhere in
  the crate, so no mediaframe type reaches this crate's public API and
  none of 0.5's three breaking counts — the `Infallible` parse split,
  the opened `subtitle::TrackOrigin`, the removed
  `subtitle::Format::PgsSub` — has anywhere to land. The pin stays at
  the no-alloc tier (`rgb`, `rgb-float`, `rgb-legacy`).

## 0.3.0 — 2026-08-20

### Changed

- **`mediatime` `0.1` → `0.3`** and **`mediaframe` `0.1` → `0.4`**. Both
  cross two majors; `mediatime` is a public dependency (this crate
  re-exports `Timebase`, `Timestamp` and `TimeRange` from
  [`frame`](src/frame.rs)), so its breakage is this crate's breakage.
  - **`Timebase` is signed.** `num: u32 → i32` and
    `den: NonZeroU32 → NonZeroI32`, matching ffmpeg's `AVRational`.
    `Timebase::new`, `try_new`, `num`, `den` and the `with_*` / `set_*`
    setters all change signature, so every caller constructing a
    `Timebase` moves its denominator literal from `NonZeroU32` to
    `NonZeroI32`. `Timebase::new` additionally panics on a negative
    numerator or denominator.
  - **`Options::{with,set}_min_frames` keeps its signature** — still
    `(frames: u32, fps: Timebase)` on all five detectors — but is now
    implemented over `mediatime::Rate`, since `Timebase::frames_to_duration`
    was deleted in favour of `Rate::saturating_frames_to_duration`. `fps`
    is still read as the rate its numerator and denominator spell
    (`Timebase::new(30, 1)` is 30 fps), and the documented
    `fps.num() == 0` panic still holds.
  - **Rounding is now to nearest, ties away from zero**
    (`AV_ROUND_NEAR_INF`) wherever mediatime converts between a
    `Duration` and a tick count, or rescales between timebases. It used
    to truncate toward zero. Nothing in this crate asks for a different
    answer — the conversions are the same rationals — but a result that
    was previously rounded down can now come back one unit larger:
    - `min_frames` durations differ by at most 1 ns
      (`1 frame @ 29.97` is `33_366_667 ns`, was `33_366_666`);
    - the cascade's periodic keyframe window differs by at most 1 tick
      under a timebase whose ticks do not divide the interval
      (4 s under `1001/30000` is 120 ticks, was 119);
    - `Timestamp::rescale_to` — used to place a mixed-timebase
      threshold cut and to anchor a shot's start pts — rounds to the
      nearest tick of the target rather than down.
    Both of this crate's pinned `min_frames` expectations
    (15 frames at 30 fps and at 29.97) land on exact values and are
    unchanged.
  - **A degenerate timebase (`num == 0`) now panics** where it used to
    answer zero. `Timestamp::saturating_sub_duration` — the virtual-past
    seed each detector takes when `initial_cut` is set — panics instead
    of silently not moving. [`cascade`](src/cascade/mod.rs) is unaffected
    (`Frames::try_new` rejects a zero-numerator timebase at admission);
    the five standalone detectors have no such guard, so a caller feeding
    a degenerate timebase to `process` now sees a panic rather than a
    no-op.
- **`mediaframe`** is bumped for graph coherence only. The `mediaframe`
  feature carries no code yet — `frame::mediaframe` is still commented
  out — and resolves mediaframe at its no-alloc tier (`rgb`,
  `rgb-float`, `rgb-legacy`), where 0.3's `Other(SmolStr)` extension
  does not exist.

## 0.2.0 — 2026-05-15

### Added

- **`keyframe` module** — per-frame quality metrics + a selection state
  machine that emits one best-frame timestamp per shot.
  - Detectors: [`luma`](src/keyframe/luma.rs),
    [`clipping`](src/keyframe/clipping.rs),
    [`saturation`](src/keyframe/saturation.rs),
    [`sharpness`](src/keyframe/sharpness.rs) (Tenengrad),
    [`noise`](src/keyframe/noise.rs) (Immerkaer),
    [`motion_blur`](src/keyframe/motion_blur.rs) (gradient anisotropy),
    [`colorfulness`](src/keyframe/colorfulness.rs) (Hasler-Süßstrunk).
  - [`metrics::FrameMetrics`](src/keyframe/metrics.rs) bundles all the
    per-frame measurements.
  - [`select`](src/keyframe/select.rs) — composite-argmax selector with
    adaptive hard-gate floors, fallback recovery on shots where every
    candidate fails the floor, and bucketed time-distance weighting.
  - [`preprocess`](src/keyframe/preprocess.rs) —
    `Downscaler` / `LumaConverter` / `HsvConverter` shared per-frame
    preprocessing utilities that own scratch buffers and hand out
    borrowed frames.
- **`frame::convert` module** — public packed-pixel → planar conversion
  routines: `bgr_to_luma` / `bgr_to_hsv_planes` (legacy BGR-only shims)
  plus the order-aware `*_with_order` variants.
- **`frame::ChannelOrder`** enum — selects whether packed 24-bit input
  is interpreted as BGR or RGB. Threaded through every kernel that
  reads packed pixels (colorfulness, luma converter, HSV converter,
  saturation detector); all detector `Options` carry a `channel_order`
  field, forward-compatible with existing serialised payloads via
  `#[serde(default)]`.
- **x86 SIMD ladder expanded** — runtime dispatch now picks the best of
  AVX2 → SSE4.1 → SSSE3 → scalar (was AVX2 → SSSE3 → scalar). The new
  SSE4.1 tier replaces `_mm_unpacklo_epi8(v, zero)` with
  `_mm_cvtepu8_epi16(v)` where applicable.
- **New SIMD kernels** on every backend (SSSE3 / SSE4.1 / AVX2 / NEON /
  wasm `simd128`): `noise` (Immerkaer Laplacian σₙ), `colorfulness`
  (single-pass Welford moments on `rg` / `yb`), and
  `gradient_anisotropy` (4-bin direction histogram with `u128` total
  reducer to keep the score inside its documented `[0, 1]` contract on
  pathological inputs).

### Changed

- **`Detector::observe()`** in the cut-detector modules now surfaces
  failures via `Result<(), ObserveError>` instead of panicking; callers
  must adapt.
- **MSRV** is now `1.95.0` (was unpinned).
- **`mediatime`** dependency unpinned from `0.1.4` → `0.1` so the crate
  picks up the `0.1.5` infallible-duration API. `Detector::finalize_shot`
  short-circuits on `Duration::is_zero()` instead of the previous
  `Option<Duration>::None` arm.
- **CI** clippy job now runs with `cargo hack clippy --each-feature -- -D warnings`
  so any lint regression fails the build.

### Internal

- Hand-written SIMD now lives under `src/arch/` (previously
  `src/content/arch/`), since the kernels are shared across the cut
  detectors and the new keyframe metrics.
- Shared `gradient_anisotropy` histogram reducer and shared
  `NOISE_COEFF` constant extracted to the `arch::` module level so the
  six backends cannot drift.

## 0.1.0 — 2026-04-XX

Initial release. Sans-I/O cut detectors:
[`histogram`](src/histogram.rs), [`phash`](src/phash.rs),
[`threshold`](src/threshold.rs), [`content`](src/content.rs),
[`adaptive`](src/adaptive.rs). Hand-written SIMD backends for aarch64
NEON, x86 SSSE3 + AVX2 (runtime-dispatched under `feature = "std"`),
and wasm `simd128`.
