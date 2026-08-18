# Changelog

All notable changes to this crate are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the project adheres to [Semantic Versioning](https://semver.org/).

## 0.3.0 — unreleased

### Changed

- **`mediatime` `0.1` → `0.3`** and **`mediaframe` `0.1` → `0.3`**. Both
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
