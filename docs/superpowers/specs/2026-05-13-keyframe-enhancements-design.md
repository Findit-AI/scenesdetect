# Keyframe Selection Enhancement — Design

> **Issue**: [#5](https://github.com/Findit-AI/scenesdetect/issues/5)
> **Branch**: `feat/keyframe-detectors`
> **Status**: Approved for plan-writing
> **Date**: 2026-05-13

## 1. Goals

Improve the keyframe-selection algorithm along two orthogonal axes:

1. **More quality signal per frame.** Today's `FrameScore` carries five
   metrics, of which only one — Tenengrad sharpness — drives the per-bucket
   winner. Three additional cheap metrics (noise, motion blur,
   colorfulness) extend the signal space.
2. **Better use of the signal we have.** The selector currently picks the
   *sharpest* eligible frame. It should pick the *best* eligible frame, as
   defined by a weighted composite of the available metrics. The current
   absolute sharpness floor also fails on legitimately low-detail shots
   (fog, night interiors); a per-shot percentile floor recovers strict
   winners in those cases instead of always degrading to fallback
   selection.

Non-goals: new perceptual-hash diversity pass, content-driven adaptive
bucket count, k-SDPP / DPP selection (all rejected in issue #5 for
documented reasons).

## 2. Background — what exists today

The keyframe module is a family of single-concern, Sans-I/O detectors:

```
src/keyframe/
├── clipping.rs       — fraction of pixels whose brightest channel is clipped
├── luma.rs           — mean + population variance of the luma plane
├── preprocess.rs     — Downscaler / LumaConverter / HsvConverter (shared scratch)
├── reduce.rs         — shared planar mean/variance kernel
├── saturation.rs     — population variance of HSV S-plane
├── score.rs          — FrameScore: the 5-field bundle (RENAMED in §3)
├── select.rs         — buffer → bucket → emit timestamps
└── sharpness.rs      — Tenengrad on 3×3 Sobel
```

Each detector has the same shape: `Options::new() / with_simd(bool)`,
`Detector::new(opts) / observe_*(...) / clear()`. Scalar today; SIMD hook
reserved. New modules in this design match that shape exactly.

The selector (`select.rs`) buffers `(Timestamp, FrameMetrics)` pairs and,
on `finalize_shot(range)`:

1. Drops stale entries (before the shot started).
2. Partitions the shot into `N` time-equal buckets (`N = ceil(duration /
   target_interval)`, clamped to `[1, max_frames_per_shot]`).
3. Per bucket, tracks a *strict* running argmax (frames passing all hard
   gates) and a *fallback* running argmax (all frames regardless).
4. Emits the strict winner per bucket, or the fallback if no strict
   winner.

Current strict-pass argmax key: `frame.sharpness`. Current strict floor:
absolute `min_sharpness = 100.0`.

## 3. `FrameMetrics` — rename, extension, and field encapsulation

`FrameScore` is renamed to `FrameMetrics`, and the module
`keyframe::score` is renamed to `keyframe::metrics`. Rationale: the struct
holds the **inputs** to scoring; the scalar **score** is what the selector
derives from those inputs via `composite_quality()`. Keeping the names
distinct sharpens the model.

**All fields are private.** Read access goes through getter methods;
mutation goes through `set_*` setters and `with_*` consuming builders.
This brings `FrameMetrics` in line with the rest of the crate (and the
project-wide convention) and removes the existing struct's anomalous
public-field exposure as part of the same change. `with_*` builders are
`const fn` (field assignment only); getters are `const fn`. Setters are
non-const because they take `&mut self`.

```rust
// src/keyframe/metrics.rs (new path)
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FrameMetrics {
    // private — accessed via methods below
    sharpness: f32,
    brightness: f32,
    luma_variance: f32,
    saturation_variance: f32,
    clipping: f32,
    noise: f32,
    motion_blur: f32,
    colorfulness: f32,
}

impl FrameMetrics {
    pub const fn new() -> Self { /* = Default::default() — written longhand for const fn */ }

    // Getters (const fn).
    pub const fn sharpness(&self) -> f32           { self.sharpness }
    pub const fn brightness(&self) -> f32          { self.brightness }
    pub const fn luma_variance(&self) -> f32       { self.luma_variance }
    pub const fn saturation_variance(&self) -> f32 { self.saturation_variance }
    pub const fn clipping(&self) -> f32            { self.clipping }
    pub const fn noise(&self) -> f32               { self.noise }
    pub const fn motion_blur(&self) -> f32         { self.motion_blur }
    pub const fn colorfulness(&self) -> f32        { self.colorfulness }

    // Consuming builders (const fn).
    pub const fn with_sharpness(mut self, v: f32) -> Self           { self.sharpness = v; self }
    pub const fn with_brightness(mut self, v: f32) -> Self          { self.brightness = v; self }
    pub const fn with_luma_variance(mut self, v: f32) -> Self       { self.luma_variance = v; self }
    pub const fn with_saturation_variance(mut self, v: f32) -> Self { self.saturation_variance = v; self }
    pub const fn with_clipping(mut self, v: f32) -> Self            { self.clipping = v; self }
    pub const fn with_noise(mut self, v: f32) -> Self               { self.noise = v; self }
    pub const fn with_motion_blur(mut self, v: f32) -> Self         { self.motion_blur = v; self }
    pub const fn with_colorfulness(mut self, v: f32) -> Self        { self.colorfulness = v; self }

    // Mutating setters (return &mut Self for chaining).
    pub fn set_sharpness(&mut self, v: f32) -> &mut Self           { self.sharpness = v; self }
    pub fn set_brightness(&mut self, v: f32) -> &mut Self          { self.brightness = v; self }
    pub fn set_luma_variance(&mut self, v: f32) -> &mut Self       { self.luma_variance = v; self }
    pub fn set_saturation_variance(&mut self, v: f32) -> &mut Self { self.saturation_variance = v; self }
    pub fn set_clipping(&mut self, v: f32) -> &mut Self            { self.clipping = v; self }
    pub fn set_noise(&mut self, v: f32) -> &mut Self               { self.noise = v; self }
    pub fn set_motion_blur(&mut self, v: f32) -> &mut Self         { self.motion_blur = v; self }
    pub fn set_colorfulness(&mut self, v: f32) -> &mut Self        { self.colorfulness = v; self }
}
```

`#[derive(Default)]` is preserved. The pipeline pattern becomes:

```rust
let stats = luma_det.observe_luma(luma);
let metrics = FrameMetrics::new()
    .with_sharpness(sharpness.observe_luma(luma))
    .with_brightness(stats.mean())
    .with_luma_variance(stats.variance())
    .with_clipping(clipping.observe_rgb(small_bgr))
    .with_saturation_variance(sat.observe_hsv(hsv))
    .with_noise(noise.observe_luma(luma))
    .with_motion_blur(motion_blur.observe_luma(luma))
    .with_colorfulness(colorfulness.observe_rgb(small_bgr));
selector.observe(ts, metrics);
```

Internal selector code uses getters (e.g. `metrics.sharpness()`) where
it currently uses field access (`score.sharpness`). Existing tests that
construct `FrameScore { sharpness, brightness, … }` literals migrate to
chained `with_*` builders. This is a single mechanical pass across the
crate.

**Breaking-change impact**: every existing reader/writer of the struct
must be updated. The crate is 0.x; the change is local to the
`keyframe::select` module and its tests inside this repository. External
downstream callers (if any) require the same mechanical migration —
documented in CHANGELOG.

### 3.1 `LumaStats` — encapsulate alongside

`keyframe::luma::LumaStats` is the only other type in the crate that
exposes `pub` fields (`mean`, `variance`). It sits directly in the
keyframe data flow — the caller wiring in §6 reads it to populate
`FrameMetrics`. To keep the convention consistent across the crate (and
avoid mixing two styles in adjacent calls inside the same pipeline),
encapsulate it in the same PR.

```rust
// src/keyframe/luma.rs (modified)
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LumaStats {
    mean: f32,
    variance: f32,
}

impl LumaStats {
    pub const fn new() -> Self                              { Self { mean: 0.0, variance: 0.0 } }
    pub const fn mean(&self) -> f32                         { self.mean }
    pub const fn variance(&self) -> f32                     { self.variance }
    pub const fn with_mean(mut self, v: f32) -> Self        { self.mean = v; self }
    pub const fn with_variance(mut self, v: f32) -> Self    { self.variance = v; self }
    pub fn set_mean(&mut self, v: f32) -> &mut Self         { self.mean = v; self }
    pub fn set_variance(&mut self, v: f32) -> &mut Self     { self.variance = v; self }
}
```

Internal construction in `keyframe::luma`'s scalar reducer switches from
`LumaStats { mean, variance }` to
`LumaStats::new().with_mean(mean).with_variance(variance)`. Callers
switch field reads to method calls.

## 4. New detectors

Each new module follows the existing detector contract verbatim:
`Options` with `with_simd`, `Detector::new`, `observe_*`, `clear`. All
output `f32`. Scalar kernels ship in v1; SIMD kernels are deferred to
follow-up PRs and gated by the existing `use_simd` flag plumbing.

**Encapsulation convention** (applies to every new struct in §§4–5): all
fields are private. Read access via `const fn`-getter methods named
after the field; mutation via `with_*` consuming builders (`const fn`
where the body is field assignment only) and `&mut self -> &mut Self`
`set_*` setters. The `Options` skeletons shown below as
`/* use_simd */` are this pattern in shorthand — concretely:

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Options { use_simd: bool }

impl Default for Options { fn default() -> Self { Self::new() } }

impl Options {
    pub const fn new() -> Self                       { Self { use_simd: true } }
    pub const fn with_simd(mut self, on: bool) -> Self { self.use_simd = on; self }
    pub const fn use_simd(&self) -> bool             { self.use_simd }
}
```

This matches the existing `keyframe::sharpness::Options`,
`keyframe::luma::Options`, etc.

### 4.1 `keyframe::noise` — Immerkaer σₙ

Immerkaer (1996) "Fast Noise Variance Estimation": one convolution of
the luma plane with the 3×3 Laplacian-of-difference mask `N`, followed
by an absolute-value reduction:

```
        ⎡  1  -2   1 ⎤
    N = ⎢ -2   4  -2 ⎥
        ⎣  1  -2   1 ⎦

    σₙ ≈ √(π/2) · (1 / (6·(W−2)·(H−2))) · Σ |luma ⊛ N|
```

Border pixels are excluded (matching the Tenengrad interior-pixel
convention). Frames with `W < 3` or `H < 3` return `0.0`. Memory access
pattern: identical to Tenengrad; 9-tap 3×3 read per interior pixel,
single i64 accumulator. SIMD-friendly. Why this and not "Laplacian MAD"
as the proposal first suggested: σₙ is O(n) single-pass; MAD requires
a sort or histogram (~3× cost) for no quality advantage on the typical
noise distribution in compressed video.

API:

```rust
pub struct Options { /* use_simd */ }
pub struct Detector { /* opts */ }
impl Detector {
    pub fn observe_luma(&mut self, luma: LumaFrame<'_>) -> f32; // σₙ in 0-255 space
}
```

Per-frame cost on 256×144 luma: ~0.10ms scalar.

### 4.2 `keyframe::motion_blur` — gradient anisotropy

Reuses `crate::arch::sobel(luma, mag, dir, w, h, use_simd)` (existing
public-crate kernel; produces i32 L1 magnitude and u8 quantized direction
∈ {0,1,2,3} for {0°, 45°, 90°, 135°}). Detector owns the two scratch
buffers and grows them monotonically.

After Sobel, build a 4-bin magnitude-weighted histogram of direction:

```
hist[k] = Σ_{(x,y) ∈ interior} mag[x,y]  where dir[x,y] == k
total   = Σ hist[k]
```

Anisotropy score:

```
anisotropy = max(0, (max(hist) / total) − 0.25) / 0.75   // ∈ [0, 1]
```

`0.25` is the uniform-direction expectation under the 4-bin
quantization; the result is `0.0` when magnitude is evenly distributed
and `1.0` when concentrated in a single direction. `total == 0`
(uniform luma) returns `0.0`. Frames < 3×3 return `0.0`.

Sobel runs twice per frame in this design — once inside Tenengrad,
once in `motion_blur`. The shared-Sobel optimisation (fused module
producing mag+dir consumed by both) is deferred. Cost of the duplicated
Sobel pass: ~0.08ms on 256×144.

API:

```rust
pub struct Options { /* use_simd */ }
pub struct Detector { /* opts + Vec<i32> mag scratch + Vec<u8> dir scratch */ }
impl Detector {
    pub fn observe_luma(&mut self, luma: LumaFrame<'_>) -> f32; // anisotropy ∈ [0, 1]
    pub fn observe_sobel(&mut self, mag: &[i32], dir: &[u8], w: usize, h: usize) -> f32;
}
```

The `observe_sobel` entry point is provided for callers willing to
share Sobel output (forward compatibility with the shared-Sobel
optimisation).

Per-frame cost on 256×144: ~0.10ms scalar (~0.08 Sobel + ~0.02
histogram + reduction).

### 4.3 `keyframe::colorfulness` — Hasler-Süßstrunk

Hasler & Süßstrunk (2003) "Measuring colourfulness in natural images":

```
rg = R − G
yb = ½(R + G) − B
σ_rgyb = √(σ²_rg + σ²_yb)
μ_rgyb = √(μ²_rg + μ²_yb)
C      = σ_rgyb + 0.3·μ_rgyb
```

Single pass over packed 24-bit BGR (downscaled). Streaming moments
(Welford-style) avoid a second pass for variance. The `rg`/`yb` channels
are computed on the fly per pixel; no intermediate planes allocated.

Byte order: the implementation treats the packed input as BGR (matching
`crate::keyframe::preprocess`). Swapping B and R changes `yb` but not
`rg` or the final `C` magnitude meaningfully — the metric is robust to
the convention. The detector documents BGR input.

API:

```rust
pub struct Options { /* use_simd */ }
pub struct Detector { /* opts */ }
impl Detector {
    pub fn observe_rgb(&mut self, rgb: RgbFrame<'_>) -> f32; // C, typical range [0, 200]
}
```

Per-frame cost on 256×144 BGR: ~0.05ms scalar.

### 4.4 `arch` plumbing

Three new dispatch fns added to `src/arch.rs`:

```rust
pub(crate) fn noise(luma: &[u8], w, h, stride, use_simd: bool) -> f32;
pub(crate) fn gradient_anisotropy(mag: &[i32], dir: &[u8], w, h, use_simd: bool) -> f32;
pub(crate) fn colorfulness(bgr: &[u8], w, h, stride, use_simd: bool) -> f32;
```

Scalar impls in `arch::scalar::Scalar`. SIMD backends deferred. The
existing dispatch convention (NEON / SSSE3 / AVX2 / wasm-simd128 ladder
under `is_x86_feature_detected!` for x86 std builds, compile-time
gating elsewhere) is preserved for when SIMD kernels are added.

## 5. Selector changes (`keyframe::select`)

### 5.1 Composite-quality argmax

Today, per bucket:

```rust
if best_strict.is_none_or(|(_, s)| sharper(score.sharpness, s)) {
    best_strict = Some((ts, score.sharpness));
}
```

Replace with:

```rust
let q = composite_quality(&metrics, &opts.weights);
if best_strict.is_none_or(|(_, s)| sharper(q, s)) {
    best_strict = Some((ts, q));
}
```

`composite_quality` (lives in `keyframe::select`; `FrameMetrics` fields
are accessed via the public getters since the struct lives in
`keyframe::metrics`):

```rust
fn composite_quality(m: &FrameMetrics, w: &CompositeWeights) -> f32 {
    w.sharpness    * (m.sharpness()    / w.sharpness_norm)
  - w.noise        * (m.noise()        / w.noise_norm)
  + w.colorfulness * (m.colorfulness() / w.colorfulness_norm)
  - w.clipping     *  m.clipping()
  - w.motion_blur  *  m.motion_blur()
}
```

`CompositeWeights` fields are private to its module (see §5.4); the
example reads `w.sharpness` because the helper is co-located with the
struct definition.

Defaults (chosen so a "good" baseline frame scores ≈ 1.0):

| Term         | Weight | Normaliser |
|--------------|-------:|-----------:|
| sharpness    |    1.0 |     1000.0 |
| noise        |    0.3 |       20.0 |
| colorfulness |    0.2 |       50.0 |
| clipping     |    0.5 |        — (already 0–1) |
| motion_blur  |    0.0 |        — (already 0–1; weight = 0 → no effect by default) |

Normalisers chosen against observed typical ranges at 256-px downscale:
Tenengrad ∈ [100, 5000], σₙ ∈ [0, 50], C ∈ [0, 200]. They are explicit
knobs, not hard-coded, so the caller can recalibrate after telemetry.
Setting `w.noise = w.colorfulness = w.motion_blur = 0.0` and
`w.clipping = 0.0` collapses the composite to pure
sharpness/sharpness_norm — identical ranking to today.

**Fallback path is unchanged**: still ranks by raw `sharpness` so
"least bad" is well-defined when every frame fails the strict gate.

### 5.2 Adaptive per-shot sharpness floor

After draining stale entries and before the bucketing walk in
`finalize_shot`, compute the in-shot 25th-percentile sharpness when at
least `min_samples` frames are buffered:

```rust
let effective_min_sharpness = if opts.adaptive_floor && in_shot.len() >= opts.min_samples {
    let mut s: Vec<f32> = in_shot.iter().map(|(_, m)| m.sharpness).collect();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let p25 = s[s.len() / 4];
    opts.min_sharpness.min(p25)         // never raise the floor
} else {
    opts.min_sharpness
};
```

The floor is only **lowered**, never raised — adaptive floor cannot make
a previously-passing frame fail. Short shots (`< min_samples`) get the
absolute floor unchanged.

Sort cost: `O(n log n)` where `n` ≤ buffered-frames-per-shot. At default
scoring rate (one score per decoded frame) and `target_interval = 4s`,
`n` is in the hundreds. The percentile pass is sub-millisecond per shot,
well below the per-shot budget.

Configurable via `with_adaptive_floor(bool)`,
`with_adaptive_floor_percentile(f32)`,
`with_adaptive_floor_min_samples(usize)`. Defaults: `true`, `0.25`, `20`.

### 5.3 Motion-blur hard gate (opt-in)

In `hard_gate`:

```rust
if opts.motion_blur_gate && m.motion_blur > opts.max_motion_blur {
    return true;
}
```

Defaults: `motion_blur_gate = false`, `max_motion_blur = 0.75`. The gate
is opt-in because gradient anisotropy on a 256-px downscale confounds
"motion blur" with "scene with strong dominant gradient direction"
(building façades, forest canopies, ocean horizons). When telemetry
validates the metric on production footage, defaults may be revisited.

### 5.4 `Options` surface

New builder methods. All preserve today's behaviour by default except
the composite-argmax replacement itself (whose defaults are tuned to
match today's ranking on "ordinary" frames):

```rust
// nested struct to avoid Options bloat; same private-fields + accessors
// convention as the rest of the crate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompositeWeights {
    sharpness:        f32,  sharpness_norm:    f32,
    noise:            f32,  noise_norm:        f32,
    colorfulness:     f32,  colorfulness_norm: f32,
    clipping:         f32,
    motion_blur:      f32,
}

impl Default for CompositeWeights { fn default() -> Self { /* §5.1 defaults */ } }

impl CompositeWeights {
    pub const fn new() -> Self;                                       // = Default, hand-written for const fn

    // Builders (paired weight + normaliser where applicable).
    pub const fn with_sharpness(self, w: f32, norm: f32) -> Self;
    pub const fn with_noise(self, w: f32, norm: f32) -> Self;
    pub const fn with_colorfulness(self, w: f32, norm: f32) -> Self;
    pub const fn with_clipping(self, w: f32) -> Self;
    pub const fn with_motion_blur(self, w: f32) -> Self;

    // Getters.
    pub const fn sharpness(&self) -> f32           { self.sharpness }
    pub const fn sharpness_norm(&self) -> f32      { self.sharpness_norm }
    // … one getter per field
}

impl Options {
    pub fn with_composite_weights(self, w: CompositeWeights) -> Self;
    pub fn with_adaptive_floor(self, on: bool) -> Self;
    pub fn with_adaptive_floor_percentile(self, p: f32) -> Self;  // panics outside [0, 1]
    pub fn with_adaptive_floor_min_samples(self, n: usize) -> Self;
    pub fn with_motion_blur_gate(self, on: bool) -> Self;
    pub fn with_max_motion_blur(self, m: f32) -> Self;            // panics outside [0, 1]

    // + matching read-only accessors (composite_weights, adaptive_floor,
    //   adaptive_floor_percentile, adaptive_floor_min_samples,
    //   motion_blur_gate, max_motion_blur).
}
```

## 6. Caller wiring (data flow)

The pipeline composes existing + new detectors per the Sans-I/O
contract:

```
BGR frame ─→ Downscaler ─→ small BGR ─┐
                                      ├─→ LumaConverter ─→ luma ─┬─→ sharpness::Detector    ─→ .with_sharpness(...)
                                      │                          ├─→ luma::Detector         ─→ stats.mean()/variance() → .with_brightness/luma_variance(...)
                                      │                          ├─→ noise::Detector        ─→ .with_noise(...)            [NEW]
                                      │                          └─→ motion_blur::Detector  ─→ .with_motion_blur(...)      [NEW]
                                      │
                                      ├─→ HsvConverter ─→ hsv ─→ saturation::Detector       ─→ .with_saturation_variance(...)
                                      │
                                      ├─→ clipping::Detector                                ─→ .with_clipping(...)
                                      └─→ colorfulness::Detector                            ─→ .with_colorfulness(...)    [NEW]

selector.observe(ts, FrameMetrics::new().with_*(…))   per frame
selector.finalize_shot(range)                         on shot boundary
```

No internal coupling between the keyframe module and other detector
families (content / histogram / phash / threshold / adaptive) is
introduced.

## 7. Error handling

No new error types. Each detector returns `f32`. Edge cases:

| Input                              | Result                                    |
|------------------------------------|-------------------------------------------|
| Frame `< 3×3`                      | `noise`, `motion_blur` return `0.0`       |
| Uniform luma                       | `motion_blur` returns `0.0` (zero total)  |
| Stride-padded source               | Padding bytes excluded (existing convention) |
| `w·h == 0` (degenerate)            | All detectors return `0.0`                |
| NaN comparison in composite_quality | `partial_cmp` returns `None` → not-greater → cannot unseat numeric incumbent (existing `sharper` convention) |

## 8. Testing strategy

### Per new detector

- Black / white / uniform-gray frame → `0.0` for noise (no high-freq
  signal) and motion_blur; `0.0` for colorfulness (zero rg/yb).
- Hand-fixture tests with deterministic inputs and pre-computed
  expected outputs (e.g. for `noise`: alternating ±k luma stripe with
  known σₙ; for `colorfulness`: red+green+blue quadrants with
  hand-computed C).
- Stride-padding-is-ignored test (mirrors existing convention).
- "Too-small frame returns zero" boundary test.
- `clear` is no-op test (parity with existing detector pattern).
- Sobel-side `observe_sobel` entry point for `motion_blur`: matches
  `observe_luma` output on equivalent input modulo float rounding.

When SIMD backends ship in a follow-up: scalar↔SIMD parity tests
guarded by `is_x86_feature_detected!` per the project's existing
test convention.

### Selector

- `composite_argmax_picks_clean_over_noisier_when_weighted`: bucket
  with `(sharp=2000, noise=15)` and `(sharp=1800, noise=3)` — the
  second wins under default weights.
- `composite_argmax_collapses_to_sharpness_when_only_sharpness_weighted`:
  zero out non-sharpness weights → identical emissions to today's
  algorithm.
- `adaptive_floor_recovers_strict_winner_in_low_detail_shot`: every
  frame has `sharpness < absolute_floor`; expect the p25-passing
  frame to emerge as strict winner instead of fallback.
- `adaptive_floor_disabled_matches_legacy_behaviour`.
- `adaptive_floor_does_not_raise_floor`: shot with high mean sharpness
  → absolute floor still applies, no change.
- `motion_blur_gate_disabled_by_default_preserves_existing_behaviour`
  across the existing test corpus.
- `motion_blur_gate_enabled_rejects_high_anisotropy_frame`.

### Regression

Existing `select.rs` tests update mechanically. The migration is two
patterns:

- Field reads (`s.brightness`) become method calls (`s.brightness()`).
- Literal constructors (`FrameScore { sharpness, brightness, … }`)
  become builder chains (`FrameMetrics::new().with_sharpness(sharpness)
  .with_brightness(brightness)…`), or, when test fixtures mutate one
  field at a time, `set_*` setters (`s.set_brightness(5.0)`).

Import paths update from `keyframe::score::FrameScore` to
`keyframe::metrics::FrameMetrics`.

## 9. Performance budget

Estimated per-frame cost at 256-px longest-side downscale (≈ 256×144 ≈
37k pixels):

| Stage                          | Existing | After |
|--------------------------------|---------:|------:|
| Downscale + colour conversions |   ~0.30  | ~0.30 |
| sharpness (Tenengrad+Sobel)    |   ~0.15  | ~0.15 |
| luma / saturation variance     |   ~0.10  | ~0.10 |
| clipping                       |   ~0.05  | ~0.05 |
| **noise** (Immerkaer σₙ)       |       —  | ~0.10 |
| **motion_blur** (Sobel+hist)   |       —  | ~0.10 |
| **colorfulness** (single pass) |       —  | ~0.05 |
| **Per-frame total**            |    ~0.60 | ~0.85 |
| **Throughput**                 |  ~1660 fps | ~1175 fps |

Composite-argmax + adaptive-floor cost is dominated by the per-shot
`O(n log n)` sort on `finalize_shot`, well below 1ms per shot for
realistic `n`.

These numbers are scalar-only. Adding SIMD backends (deferred follow-up)
recovers some of the new overhead.

## 10. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Composite weights mis-calibrated, keyframe quality regresses | Defaults preserve sharpness as the dominant term. Zero non-sharpness weights to fall back to today's ranking exactly. |
| Motion-blur metric mis-classifies directional scenes | Gate off by default; composite weight 0.0 by default. Telemetry-only out of the box. |
| Adaptive floor noisy on short shots | `min_samples = 20` threshold below which the absolute floor is used unchanged. |
| `FrameMetrics` rename + `LumaStats` field encapsulation break downstream callers | Crate is 0.x. Internal callers updated in the same PR. Public-API change documented in CHANGELOG. |
| Per-frame budget overrun on low-power targets | ~1175 fps scalar at 256-px is well above any realistic ingest rate (~120 fps for 4× real-time on 30fps source). Acceptable. |
| Doubled Sobel pass wastes work | Acceptable trade for orthogonal module design in v1. Shared-Sobel fusion noted as a future optimisation. |

## 11. Out of scope (deferred to separate work)

- SIMD kernels for `noise`, `motion_blur`, `colorfulness` (scalar-only
  in this PR; existing `use_simd` plumbing is in place for the
  follow-up).
- Shared-Sobel optimisation (sharpness + motion_blur fused into one
  Sobel pass).
- Phase 2 of issue #5 (pHash diversity refinement): contradicts the
  stated VLM-temporal-regularity goal in the same proposal.
- Phase 3 of issue #5 (content-driven adaptive bucket count): introduces
  cross-module coupling; caller-side `target_interval` overrides cover
  the use case.

(`FrameMetrics` field encapsulation is **in scope** — see §3 — not
deferred.)

## 12. References

- Immerkaer, J. (1996). "Fast Noise Variance Estimation". *Computer
  Vision and Image Understanding* 64(2): 300–302.
- Hasler, D. & Süßstrunk, S. (2003). "Measuring Colourfulness in
  Natural Images". *Proc. SPIE Human Vision and Electronic Imaging
  VIII* 5007.
- Pertuz, S., Puig, D., Garcia, M. (2013). "Analysis of focus measure
  operators for shape-from-focus" — Tenengrad / sharpness families.
- Issue #5 — original proposal, including DPP rejection rationale.
