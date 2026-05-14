# Changelog

All notable changes to this crate are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the project adheres to [Semantic Versioning](https://semver.org/).

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
