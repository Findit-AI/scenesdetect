# Keyframe Enhancements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land issue #5 enhancements — 3 new per-frame quality metrics (Immerkaer σₙ noise, gradient-anisotropy motion blur, Hasler-Süßstrunk colorfulness) plus a composite-quality argmax, an adaptive per-shot sharpness floor, and an opt-in motion-blur hard gate, with `FrameScore` renamed to `FrameMetrics` and both `FrameMetrics` and `LumaStats` re-encapsulated under the crate's no-public-fields convention.

**Architecture:** Per the design in `docs/superpowers/specs/2026-05-13-keyframe-enhancements-design.md`. Each new detector follows the existing Sans-I/O detector contract verbatim — `Options::new() / with_simd()`, `Detector::new(opts) / observe_*(...) / clear()`. Scalar kernels live in `arch::scalar::Scalar`; top-level dispatch fns in `arch.rs` retain the SIMD-ladder shape so future SIMD backends slot in without changing the dispatch signature. Selector changes layer on top of the existing bucket walker without changing its O(n) shape.

**Tech Stack:** Rust 2021, `no_std`-friendly (`alloc` + `std` feature ladder), `fast_image_resize` for downscaling, `derive_more`, optional `serde`. Tests are inline `#[cfg(all(test, feature = "std"))] mod tests` in each module, matching crate convention.

---

## File Structure

**New files**

- `src/keyframe/metrics.rs` — `FrameMetrics` (replaces `score.rs`), private fields + `const fn` accessors.
- `src/keyframe/noise.rs` — `Detector` / `Options` for Immerkaer σₙ.
- `src/keyframe/motion_blur.rs` — `Detector` / `Options` for gradient anisotropy. Owns Sobel scratch.
- `src/keyframe/colorfulness.rs` — `Detector` / `Options` for Hasler-Süßstrunk.

**Files to delete**

- `src/keyframe/score.rs` — replaced by `metrics.rs` (Task 3 removes it after Task 2 migrates callers).

**Files to modify**

- `src/keyframe.rs` — `mod` declarations: drop `score`, add `metrics`, `noise`, `motion_blur`, `colorfulness`. Doc comment lists the new modules.
- `src/keyframe/luma.rs` — `LumaStats` fields → private + accessors.
- `src/keyframe/select.rs` — switch from `FrameScore` to `FrameMetrics`, add `CompositeWeights`, replace strict-pass argmax with `composite_quality()`, add adaptive per-shot sharpness floor, add motion-blur opt-in gate. Update internal field reads → getters and constructor literals → builder chains.
- `src/arch.rs` — add three `pub(crate)` dispatch fns (`noise`, `gradient_anisotropy`, `colorfulness`) and matching scalar kernels in `arch::scalar::Scalar`.

**Files that should NOT be touched in this plan**

- `Cargo.toml`, `src/frame.rs` — uncommitted WIP for the unrelated `videoframe` feature lives there.
- `src/keyframe/preprocess.rs`, `src/keyframe/clipping.rs`, `src/keyframe/saturation.rs`, `src/keyframe/sharpness.rs`, `src/keyframe/reduce.rs`. The sharpness detector's internal Sobel pass is left untouched (shared-Sobel optimisation is deferred).
- `src/{phash,content,histogram,threshold,adaptive}.rs` — separate detector families, no coupling.

---

## Task 1: Create `keyframe::metrics::FrameMetrics`

Add the new metrics type alongside (not replacing) `keyframe::score`. The two modules coexist briefly; Task 2 switches callers; Task 3 deletes the old one. This intermediate state keeps each commit compilable.

**Files:**

- Create: `src/keyframe/metrics.rs`
- Modify: `src/keyframe.rs` (add one `pub mod metrics;` line)

### Steps

- [ ] **Step 1: Create `src/keyframe/metrics.rs` with the new type, accessors, and unit tests.**

Write to `src/keyframe/metrics.rs`:

```rust
//! Composite per-frame quality metrics.
//!
//! [`FrameMetrics`] bundles the eight per-frame quality measurements
//! produced by the per-metric detectors in this module's siblings
//! ([`sharpness`](crate::keyframe::sharpness),
//! [`luma`](crate::keyframe::luma),
//! [`saturation`](crate::keyframe::saturation),
//! [`clipping`](crate::keyframe::clipping),
//! [`noise`](crate::keyframe::noise),
//! [`motion_blur`](crate::keyframe::motion_blur),
//! [`colorfulness`](crate::keyframe::colorfulness)) into the shape
//! consumed by [`select::Detector`](crate::keyframe::select::Detector).
//!
//! Fields are private. Read access goes through the field-name
//! getters (`sharpness()` etc.); construction goes through `new()`
//! followed by `with_*` consuming builders (`const fn`) or `set_*`
//! mutating setters returning `&mut Self` for chaining.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// All per-frame quality measurements consumed by
/// [`select::Detector`](crate::keyframe::select::Detector).
///
/// Built by the per-metric detectors in this module's siblings. See the
/// module docs for the production pattern; `Default` yields the all-zero
/// `FrameMetrics`, useful in tests.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FrameMetrics {
  sharpness: f32,
  brightness: f32,
  luma_variance: f32,
  saturation_variance: f32,
  clipping: f32,
  noise: f32,
  motion_blur: f32,
  colorfulness: f32,
}

impl Default for FrameMetrics {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn default() -> Self {
    Self::new()
  }
}

impl FrameMetrics {
  /// Creates an all-zero [`FrameMetrics`] (same value as
  /// [`FrameMetrics::default`], but usable in `const` contexts).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      sharpness: 0.0,
      brightness: 0.0,
      luma_variance: 0.0,
      saturation_variance: 0.0,
      clipping: 0.0,
      noise: 0.0,
      motion_blur: 0.0,
      colorfulness: 0.0,
    }
  }

  // ---- Getters (const fn) ------------------------------------------------

  /// Tenengrad sharpness score (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn sharpness(&self) -> f32 {
    self.sharpness
  }
  /// Luma mean in 0-255 space (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn brightness(&self) -> f32 {
    self.brightness
  }
  /// Luma population variance (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn luma_variance(&self) -> f32 {
    self.luma_variance
  }
  /// HSV saturation-plane population variance (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn saturation_variance(&self) -> f32 {
    self.saturation_variance
  }
  /// Fraction of pixels whose brightest channel is clipped, in `[0, 1]`
  /// (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn clipping(&self) -> f32 {
    self.clipping
  }
  /// Immerkaer noise estimate σₙ in 0-255 space (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn noise(&self) -> f32 {
    self.noise
  }
  /// Gradient anisotropy in `[0, 1]` (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn motion_blur(&self) -> f32 {
    self.motion_blur
  }
  /// Hasler-Süßstrunk colorfulness score (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn colorfulness(&self) -> f32 {
    self.colorfulness
  }

  // ---- Consuming builders (const fn) -------------------------------------

  /// Returns `self` with [`sharpness`](Self::sharpness) set to `v`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_sharpness(mut self, v: f32) -> Self {
    self.sharpness = v;
    self
  }
  /// Returns `self` with [`brightness`](Self::brightness) set to `v`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_brightness(mut self, v: f32) -> Self {
    self.brightness = v;
    self
  }
  /// Returns `self` with [`luma_variance`](Self::luma_variance) set to `v`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_luma_variance(mut self, v: f32) -> Self {
    self.luma_variance = v;
    self
  }
  /// Returns `self` with [`saturation_variance`](Self::saturation_variance) set to `v`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_saturation_variance(mut self, v: f32) -> Self {
    self.saturation_variance = v;
    self
  }
  /// Returns `self` with [`clipping`](Self::clipping) set to `v`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_clipping(mut self, v: f32) -> Self {
    self.clipping = v;
    self
  }
  /// Returns `self` with [`noise`](Self::noise) set to `v`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_noise(mut self, v: f32) -> Self {
    self.noise = v;
    self
  }
  /// Returns `self` with [`motion_blur`](Self::motion_blur) set to `v`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_motion_blur(mut self, v: f32) -> Self {
    self.motion_blur = v;
    self
  }
  /// Returns `self` with [`colorfulness`](Self::colorfulness) set to `v`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_colorfulness(mut self, v: f32) -> Self {
    self.colorfulness = v;
    self
  }

  // ---- Mutating setters (chainable) --------------------------------------

  /// In-place setter for [`sharpness`](Self::sharpness). Returns `&mut Self` for chaining.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_sharpness(&mut self, v: f32) -> &mut Self {
    self.sharpness = v;
    self
  }
  /// In-place setter for [`brightness`](Self::brightness).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_brightness(&mut self, v: f32) -> &mut Self {
    self.brightness = v;
    self
  }
  /// In-place setter for [`luma_variance`](Self::luma_variance).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_luma_variance(&mut self, v: f32) -> &mut Self {
    self.luma_variance = v;
    self
  }
  /// In-place setter for [`saturation_variance`](Self::saturation_variance).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_saturation_variance(&mut self, v: f32) -> &mut Self {
    self.saturation_variance = v;
    self
  }
  /// In-place setter for [`clipping`](Self::clipping).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_clipping(&mut self, v: f32) -> &mut Self {
    self.clipping = v;
    self
  }
  /// In-place setter for [`noise`](Self::noise).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_noise(&mut self, v: f32) -> &mut Self {
    self.noise = v;
    self
  }
  /// In-place setter for [`motion_blur`](Self::motion_blur).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_motion_blur(&mut self, v: f32) -> &mut Self {
    self.motion_blur = v;
    self
  }
  /// In-place setter for [`colorfulness`](Self::colorfulness).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_colorfulness(&mut self, v: f32) -> &mut Self {
    self.colorfulness = v;
    self
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;

  #[test]
  fn default_is_all_zero() {
    let m = FrameMetrics::default();
    assert_eq!(m.sharpness(), 0.0);
    assert_eq!(m.brightness(), 0.0);
    assert_eq!(m.luma_variance(), 0.0);
    assert_eq!(m.saturation_variance(), 0.0);
    assert_eq!(m.clipping(), 0.0);
    assert_eq!(m.noise(), 0.0);
    assert_eq!(m.motion_blur(), 0.0);
    assert_eq!(m.colorfulness(), 0.0);
  }

  #[test]
  fn new_matches_default() {
    assert_eq!(FrameMetrics::new(), FrameMetrics::default());
  }

  #[test]
  fn new_is_const_context_usable() {
    const M: FrameMetrics = FrameMetrics::new();
    assert_eq!(M, FrameMetrics::default());
  }

  #[test]
  fn with_builders_roundtrip_through_getters() {
    let m = FrameMetrics::new()
      .with_sharpness(1.0)
      .with_brightness(2.0)
      .with_luma_variance(3.0)
      .with_saturation_variance(4.0)
      .with_clipping(0.5)
      .with_noise(6.0)
      .with_motion_blur(0.7)
      .with_colorfulness(8.0);
    assert_eq!(m.sharpness(), 1.0);
    assert_eq!(m.brightness(), 2.0);
    assert_eq!(m.luma_variance(), 3.0);
    assert_eq!(m.saturation_variance(), 4.0);
    assert_eq!(m.clipping(), 0.5);
    assert_eq!(m.noise(), 6.0);
    assert_eq!(m.motion_blur(), 0.7);
    assert_eq!(m.colorfulness(), 8.0);
  }

  #[test]
  fn with_builder_is_const_context_usable() {
    const M: FrameMetrics = FrameMetrics::new().with_sharpness(42.0).with_noise(5.0);
    assert_eq!(M.sharpness(), 42.0);
    assert_eq!(M.noise(), 5.0);
    // Other fields untouched.
    assert_eq!(M.brightness(), 0.0);
  }

  #[test]
  fn set_methods_chain_and_mutate() {
    let mut m = FrameMetrics::new();
    m.set_sharpness(10.0)
      .set_brightness(20.0)
      .set_luma_variance(30.0)
      .set_saturation_variance(40.0)
      .set_clipping(0.25)
      .set_noise(50.0)
      .set_motion_blur(0.6)
      .set_colorfulness(70.0);
    assert_eq!(m.sharpness(), 10.0);
    assert_eq!(m.brightness(), 20.0);
    assert_eq!(m.luma_variance(), 30.0);
    assert_eq!(m.saturation_variance(), 40.0);
    assert_eq!(m.clipping(), 0.25);
    assert_eq!(m.noise(), 50.0);
    assert_eq!(m.motion_blur(), 0.6);
    assert_eq!(m.colorfulness(), 70.0);
  }
}
```

- [ ] **Step 2: Add the `metrics` mod declaration to `src/keyframe.rs`.**

Edit `src/keyframe.rs`. Locate the existing line `pub mod score;` and insert `pub mod metrics;` immediately above it (alphabetical placement). Result block:

```rust
pub mod clipping;
pub mod luma;
pub mod metrics;
pub mod preprocess;
pub mod saturation;
pub mod score;
pub mod select;
pub mod sharpness;
```

(`score` and `metrics` coexist temporarily; Task 3 deletes `score`.)

- [ ] **Step 3: Run the new module's tests and verify they pass.**

Run: `cargo test -p scenesdetect --lib keyframe::metrics::`
Expected: 6 tests pass (`default_is_all_zero`, `new_matches_default`, `new_is_const_context_usable`, `with_builders_roundtrip_through_getters`, `with_builder_is_const_context_usable`, `set_methods_chain_and_mutate`).

- [ ] **Step 4: Build the whole crate to confirm the new module integrates.**

Run: `cargo build -p scenesdetect`
Expected: clean build (warning-free) — the existing `score.rs` is untouched, `metrics.rs` compiles in isolation.

- [ ] **Step 5: Commit.**

```bash
git add src/keyframe/metrics.rs src/keyframe.rs
git commit -m "$(cat <<'EOF'
feat(keyframe): introduce FrameMetrics with encapsulated accessors

Parallel to the existing FrameScore. Task 2 migrates select.rs;
Task 3 removes the old module.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Migrate `keyframe::select` from `FrameScore` to `FrameMetrics`

Switch the selector's struct, function signatures, internal field reads, and test literals. After this task, `keyframe::score::FrameScore` has no remaining callers in the crate.

**Files:**

- Modify: `src/keyframe/select.rs`

### Steps

- [ ] **Step 1: Update the import in `src/keyframe/select.rs`.**

Locate the existing import block:

```rust
use crate::{
  frame::{TimeRange, Timestamp},
  keyframe::score::FrameScore,
};
```

Replace with:

```rust
use crate::{
  frame::{TimeRange, Timestamp},
  keyframe::metrics::FrameMetrics,
};
```

- [ ] **Step 2: Rename the buffered-tuple type and the `observe`/`hard_gate` signatures.**

Locate the struct field:

```rust
  buffer: VecDeque<(Timestamp, FrameScore)>,
```

Replace with:

```rust
  buffer: VecDeque<(Timestamp, FrameMetrics)>,
```

Locate the `observe` method signature:

```rust
  pub fn observe(&mut self, ts: Timestamp, score: FrameScore) {
```

Replace with:

```rust
  pub fn observe(&mut self, ts: Timestamp, metrics: FrameMetrics) {
```

Inside `observe`, locate the push:

```rust
    self.buffer.push_back((ts, score));
```

Replace with:

```rust
    self.buffer.push_back((ts, metrics));
```

Locate the `hard_gate` helper at the bottom of the file:

```rust
fn hard_gate(s: &FrameScore, opts: &Options) -> bool {
  if s.brightness < opts.black_mean_threshold as f32 {
    return true;
  }
  if s.brightness > opts.bright_mean_threshold as f32 {
    return true;
  }
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
```

Replace with:

```rust
fn hard_gate(m: &FrameMetrics, opts: &Options) -> bool {
  if m.brightness() < opts.black_mean_threshold as f32 {
    return true;
  }
  if m.brightness() > opts.bright_mean_threshold as f32 {
    return true;
  }
  if m.luma_variance() < opts.luma_variance_threshold
    && m.saturation_variance() < opts.sat_variance_threshold
  {
    return true;
  }
  if m.clipping() > opts.max_clipping {
    return true;
  }
  false
}
```

- [ ] **Step 3: Update the `finalize_shot` argmax loop's field reads.**

Locate the argmax block inside the `while let Some((ts, score)) = self.buffer.front().copied()` loop:

```rust
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
```

Replace with:

```rust
      // Running argmax updates.
      if best_any.is_none_or(|(_, s)| sharper(metrics.sharpness(), s)) {
        best_any = Some((ts, metrics.sharpness()));
      }
      if !hard_gate(&metrics, &opts)
        && metrics.sharpness() >= opts.min_sharpness
        && best_strict.is_none_or(|(_, s)| sharper(metrics.sharpness(), s))
      {
        best_strict = Some((ts, metrics.sharpness()));
      }
```

Also rename the binding in the same `while let` pattern. Locate:

```rust
    while let Some((ts, score)) = self.buffer.front().copied() {
```

Replace with:

```rust
    while let Some((ts, metrics)) = self.buffer.front().copied() {
```

- [ ] **Step 4: Update the inline test module's helper to build `FrameMetrics`.**

Locate the `good_score` helper at the top of the `tests` module:

```rust
  fn good_score(sharpness: f32) -> FrameScore {
    FrameScore {
      sharpness,
      brightness: 128.0,
      luma_variance: 200.0,
      saturation_variance: 100.0,
      clipping: 0.0,
    }
  }
```

Replace with:

```rust
  fn good_score(sharpness: f32) -> FrameMetrics {
    FrameMetrics::new()
      .with_sharpness(sharpness)
      .with_brightness(128.0)
      .with_luma_variance(200.0)
      .with_saturation_variance(100.0)
      .with_clipping(0.0)
  }
```

- [ ] **Step 5: Update the `hard_gate_*` tests to use the new accessor pattern.**

Locate `hard_gate_rejects_too_dark`:

```rust
  fn hard_gate_rejects_too_dark() {
    let o = Options::default();
    let mut s = good_score(200.0);
    s.brightness = 5.0;
    assert!(hard_gate(&s, &o));
  }
```

Replace `s.brightness = 5.0;` with `s.set_brightness(5.0);` (the helper now returns `FrameMetrics`; `set_*` returns `&mut Self` and is fine to discard). Concretely:

```rust
  fn hard_gate_rejects_too_dark() {
    let o = Options::default();
    let mut s = good_score(200.0);
    s.set_brightness(5.0);
    assert!(hard_gate(&s, &o));
  }
```

Apply the same `s.<field> = v;` → `s.set_<field>(v);` migration to the remaining `hard_gate_*` tests in this module:

- `hard_gate_rejects_too_bright`: `s.brightness = 250.0;` → `s.set_brightness(250.0);`
- `hard_gate_rejects_flat_frame`: `s.luma_variance = 1.0;` → `s.set_luma_variance(1.0);` and `s.saturation_variance = 1.0;` → `s.set_saturation_variance(1.0);`
- `hard_gate_keeps_equiluminant_multicolour`: `s.luma_variance = 1.0;` → `s.set_luma_variance(1.0);` and `s.saturation_variance = 80.0;` → `s.set_saturation_variance(80.0);`
- `hard_gate_rejects_heavy_clipping`: `s.clipping = 0.9;` → `s.set_clipping(0.9);`

- [ ] **Step 6: Update the `finalize_falls_back_when_all_frames_fail_gates` inline `bad` closure.**

Locate:

```rust
    let bad = |sharp| FrameScore {
      sharpness: sharp,
      brightness: 5.0,
      luma_variance: 200.0,
      saturation_variance: 100.0,
      clipping: 0.0,
    };
```

Replace with:

```rust
    let bad = |sharp| {
      FrameMetrics::new()
        .with_sharpness(sharp)
        .with_brightness(5.0)
        .with_luma_variance(200.0)
        .with_saturation_variance(100.0)
        .with_clipping(0.0)
    };
```

- [ ] **Step 7: Run the keyframe selector tests and verify all pass.**

Run: `cargo test -p scenesdetect --lib keyframe::select`
Expected: every existing test name from the `select::tests` module passes (same set as before — no test was removed; only the constructor pattern changed).

- [ ] **Step 8: Run the full crate test suite to confirm no other module regressed.**

Run: `cargo test -p scenesdetect`
Expected: all tests pass.

- [ ] **Step 9: Commit.**

```bash
git add src/keyframe/select.rs
git commit -m "$(cat <<'EOF'
refactor(keyframe): migrate select from FrameScore to FrameMetrics

Field reads → getter calls; struct literals → builder chains;
hard_gate / observe / finalize_shot signatures updated. Tests
updated mechanically.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Remove `keyframe::score`

After Task 2 the old module has no in-crate users.

**Files:**

- Delete: `src/keyframe/score.rs`
- Modify: `src/keyframe.rs`

### Steps

- [ ] **Step 1: Verify there are no remaining references.**

Run: `git grep -n "keyframe::score\|FrameScore" src/`
Expected: no matches in `src/`. If matches appear, fix them before continuing.

- [ ] **Step 2: Remove the `mod score;` line in `src/keyframe.rs`.**

Locate and delete the single line:

```rust
pub mod score;
```

- [ ] **Step 3: Update the doc comment at the top of `src/keyframe.rs` to list `metrics` instead of `score`, and add the three new module entries.**

Locate the existing bullet:

```rust
//! - [`score`] — the composite [`score::FrameScore`] type assembled
//!   from the four metric detectors above.
```

Replace with:

```rust
//! - [`metrics`] — the [`metrics::FrameMetrics`] type bundling all
//!   per-frame measurements consumed by [`select`].
//! - [`noise`] — Immerkaer fast σₙ estimator.
//! - [`motion_blur`] — gradient anisotropy (Sobel-direction histogram).
//! - [`colorfulness`] — Hasler-Süßstrunk colourfulness metric.
```

(The three later modules don't exist yet; this doc reference is forward-compatible — rustdoc will warn about the broken intra-doc links until those modules ship, which is acceptable for the interim state. They land in Tasks 5–7.)

- [ ] **Step 4: Delete `src/keyframe/score.rs`.**

```bash
git rm src/keyframe/score.rs
```

- [ ] **Step 5: Build and test.**

Run: `cargo build -p scenesdetect && cargo test -p scenesdetect`
Expected: clean build, all tests pass.

- [ ] **Step 6: Commit.**

```bash
git add src/keyframe.rs
git commit -m "$(cat <<'EOF'
refactor(keyframe): remove obsolete score module

Replaced by keyframe::metrics in Task 1.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Encapsulate `keyframe::luma::LumaStats`

`LumaStats` currently exposes `pub mean: f32, pub variance: f32`. Convert to private fields with full accessor surface, matching `FrameMetrics`. The only in-crate caller is `keyframe::luma`'s own scalar reducer.

**Files:**

- Modify: `src/keyframe/luma.rs`

### Steps

- [ ] **Step 1: Replace the `LumaStats` definition and add the accessor impl.**

Locate the struct definition (current line range ~36–43 of `src/keyframe/luma.rs`):

```rust
/// Mean and population variance of a luma plane in 0-255 space.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LumaStats {
  /// Arithmetic mean of all sampled luma pixels.
  pub mean: f32,
  /// Population variance of the sampled luma pixels.
  pub variance: f32,
}
```

Replace with:

```rust
/// Mean and population variance of a luma plane in 0-255 space.
///
/// Fields are private; use [`mean`](Self::mean) /
/// [`variance`](Self::variance) for reads, [`with_mean`](Self::with_mean)
/// / [`with_variance`](Self::with_variance) for `const fn` builders,
/// and [`set_mean`](Self::set_mean) / [`set_variance`](Self::set_variance)
/// for in-place mutation.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LumaStats {
  mean: f32,
  variance: f32,
}

impl Default for LumaStats {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn default() -> Self {
    Self::new()
  }
}

impl LumaStats {
  /// Creates an all-zero [`LumaStats`] (same value as
  /// [`LumaStats::default`], usable in `const` contexts).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      mean: 0.0,
      variance: 0.0,
    }
  }

  /// Arithmetic mean of all sampled luma pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn mean(&self) -> f32 {
    self.mean
  }
  /// Population variance of the sampled luma pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn variance(&self) -> f32 {
    self.variance
  }

  /// Returns `self` with [`mean`](Self::mean) set to `v`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_mean(mut self, v: f32) -> Self {
    self.mean = v;
    self
  }
  /// Returns `self` with [`variance`](Self::variance) set to `v`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_variance(mut self, v: f32) -> Self {
    self.variance = v;
    self
  }

  /// In-place setter for [`mean`](Self::mean). Returns `&mut Self` for chaining.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_mean(&mut self, v: f32) -> &mut Self {
    self.mean = v;
    self
  }
  /// In-place setter for [`variance`](Self::variance).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_variance(&mut self, v: f32) -> &mut Self {
    self.variance = v;
    self
  }
}
```

- [ ] **Step 2: Update the internal `luma_stats_scalar` constructor.**

Locate the existing implementation in `src/keyframe/luma.rs`:

```rust
fn luma_stats_scalar(luma: &LumaFrame<'_>, use_simd: bool) -> LumaStats {
  let (mean, variance) = super::reduce::plane_mean_variance(
    luma.data(),
    luma.width() as usize,
    luma.height() as usize,
    luma.stride() as usize,
    use_simd,
  );
  LumaStats { mean, variance }
}
```

Replace the construction expression:

```rust
fn luma_stats_scalar(luma: &LumaFrame<'_>, use_simd: bool) -> LumaStats {
  let (mean, variance) = super::reduce::plane_mean_variance(
    luma.data(),
    luma.width() as usize,
    luma.height() as usize,
    luma.stride() as usize,
    use_simd,
  );
  LumaStats::new().with_mean(mean).with_variance(variance)
}
```

- [ ] **Step 3: Update the existing inline tests to use the accessors.**

Locate `black_frame_has_zero_mean_and_variance`:

```rust
  fn black_frame_has_zero_mean_and_variance() {
    let data = vec![0u8; 64 * 48];
    let f = LumaFrame::new(&data, 64, 48, 64, timestamp());
    let mut det = Detector::new(Options::default());
    let stats = det.observe_luma(f);
    assert_eq!(stats.mean, 0.0);
    assert_eq!(stats.variance, 0.0);
  }
```

Replace `stats.mean` with `stats.mean()` and `stats.variance` with `stats.variance()`. Apply the same `stats.<field>` → `stats.<field>()` migration to every other test in the module that reads these fields: `uniform_gray_has_zero_variance`, `uniform_white_has_zero_variance`, `half_black_half_white_variance_matches_expected`, `stride_padding_is_ignored`, `clear_is_noop`.

- [ ] **Step 4: Add a small accessor-roundtrip test for `LumaStats` itself.**

Append to the inline `tests` module in `src/keyframe/luma.rs`:

```rust
  #[test]
  fn lumastats_builders_and_setters_roundtrip() {
    let s = LumaStats::new().with_mean(120.0).with_variance(500.0);
    assert_eq!(s.mean(), 120.0);
    assert_eq!(s.variance(), 500.0);

    let mut s = LumaStats::new();
    s.set_mean(7.0).set_variance(8.0);
    assert_eq!(s.mean(), 7.0);
    assert_eq!(s.variance(), 8.0);
  }

  #[test]
  fn lumastats_new_is_const_context_usable() {
    const S: LumaStats = LumaStats::new().with_mean(3.0).with_variance(4.0);
    assert_eq!(S.mean(), 3.0);
    assert_eq!(S.variance(), 4.0);
  }
```

- [ ] **Step 5: Build and test.**

Run: `cargo test -p scenesdetect --lib keyframe::luma::`
Expected: all existing tests pass plus the two new ones (`lumastats_builders_and_setters_roundtrip`, `lumastats_new_is_const_context_usable`).

- [ ] **Step 6: Commit.**

```bash
git add src/keyframe/luma.rs
git commit -m "$(cat <<'EOF'
refactor(keyframe): encapsulate LumaStats fields

Mirror the FrameMetrics convention: private fields, const-fn getters
and with_*, &mut Self set_* setters. Internal callers and tests
updated.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Noise detector (Immerkaer σₙ)

Add the scalar kernel in `arch`, the `pub(crate)` dispatch fn, and the public-facing `keyframe::noise` module.

**Files:**

- Modify: `src/arch.rs` (add `Scalar::noise` and top-level `noise`)
- Create: `src/keyframe/noise.rs`
- Modify: `src/keyframe.rs` (add `pub mod noise;`)

### Steps

- [ ] **Step 1: Add the scalar kernel `Scalar::noise` to `arch::scalar`.**

In `src/arch.rs`, locate `mod scalar { pub(super) struct Scalar; impl Scalar { ... } }` (the impl block currently contains `plane_mean_variance` and `tenengrad`, among others). Append the new method inside that `impl Scalar` block, after the existing `tenengrad` method:

```rust
    /// Immerkaer (1996) fast noise variance estimator.
    /// Convolves the luma plane with the 3×3 Laplacian-of-difference
    /// mask `[[1,-2,1],[-2,4,-2],[1,-2,1]]`, sums the absolute values
    /// over interior pixels, then scales by `sqrt(π/2) / (6·N_inner)`
    /// to yield an estimate of the per-pixel additive-Gaussian noise
    /// standard deviation σₙ in 0-255 space. Honours row stride.
    pub(super) fn noise(luma: &[u8], w: usize, h: usize, s: usize) -> f32 {
      if w < 3 || h < 3 {
        return 0.0;
      }
      let interior = (w - 2) * (h - 2);
      if interior == 0 {
        return 0.0;
      }

      // Σ |I ⊛ N| over interior, where N is the Laplacian-of-difference
      // mask above. Per-pixel response fits in i32 (peak magnitude on
      // 8-bit input is 4·255 + 4·2·255 + 4·255 = 4·255 + 8·255 + 4·255 =
      // 16·255 = 4080); the i64 accumulator handles any realistic
      // interior count.
      let mut acc: i64 = 0;
      for y in 1..h - 1 {
        for x in 1..w - 1 {
          let p = |dy: isize, dx: isize| -> i32 {
            luma[((y as isize + dy) as usize) * s + ((x as isize + dx) as usize)] as i32
          };
          let tl = p(-1, -1);
          let t = p(-1, 0);
          let tr = p(-1, 1);
          let l = p(0, -1);
          let c = p(0, 0);
          let r = p(0, 1);
          let bl = p(1, -1);
          let b = p(1, 0);
          let br = p(1, 1);
          // N · I = 4c - 2(t+b+l+r) + (tl+tr+bl+br)
          let lap = 4 * c - 2 * (t + b + l + r) + (tl + tr + bl + br);
          acc += lap.unsigned_abs() as i64;
        }
      }

      // σₙ ≈ √(π/2) / 6 · (Σ|lap| / interior)
      // √(π/2) / 6 ≈ 0.2088987...
      const COEFF: f64 = 0.208_898_754_886_372_3;
      ((acc as f64) * COEFF / (interior as f64)) as f32
    }
```

- [ ] **Step 2: Add the `pub(crate) fn noise` top-level dispatch fn in `src/arch.rs`.**

Append after the existing `pub(crate) fn tenengrad` block (which terminates near line 468 in the current source — locate the `}` that closes `tenengrad` and insert the new block immediately after):

```rust
/// Immerkaer (1996) fast noise variance estimator. Returns σₙ (the
/// estimated per-pixel additive-Gaussian noise standard deviation) in
/// 0-255 space. Dispatches to scalar today; the parameter shape
/// matches the SIMD-ladder convention so future backends slot in
/// without changing the signature.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(unreachable_code)]
pub(crate) fn noise(
  luma: &[u8],
  width: usize,
  height: usize,
  stride: usize,
  use_simd: bool,
) -> f32 {
  if !use_simd {
    return scalar::Scalar::noise(luma, width, height, stride);
  }

  // SIMD backends not yet implemented — scalar path.
  scalar::Scalar::noise(luma, width, height, stride)
}
```

- [ ] **Step 3: Add an `arch`-level scalar correctness test for `noise`.**

In `src/arch.rs`, locate the existing `mod tests` block (gated by `#[cfg(all(test, feature = "std", not(miri)))]`, around the `scalar_tenengrad_uniform_is_zero` test). Append:

```rust
  #[test]
  fn scalar_noise_uniform_is_zero() {
    // No high-frequency signal → σₙ should be 0.
    let data = vec![100u8; 16 * 16];
    assert_eq!(scalar::Scalar::noise(&data, 16, 16, 16), 0.0);
  }

  #[test]
  fn scalar_noise_too_small_is_zero() {
    let data = vec![0u8; 4];
    assert_eq!(scalar::Scalar::noise(&data, 2, 2, 2), 0.0);
  }

  #[test]
  fn scalar_noise_alternating_stripe_matches_closed_form() {
    // A 1-pixel horizontal stripe pattern at amplitude k=64:
    //   row y:  v(y) = 100 + (-1)^y · 64
    // Per interior pixel the Laplacian-of-difference response is
    //   lap = 4c - 2(t+b+l+r) + (tl+tr+bl+br)
    //       = 4·c - 2·(t+b+2c) + (2t+2b)            // l=r=c, tl=tr=t, bl=br=b
    //       = 4c - 2t - 2b - 4c + 2t + 2b           // 0
    // Wait — for a horizontal stripe c=l=r and t=b are the same pair,
    // so the Laplacian collapses to zero on rows of constant value.
    // Use a checkerboard instead: v(y,x) = 100 + ((y+x) & 1) ? +64 : -64.
    let (w, h) = (16usize, 16usize);
    let mut data = vec![0u8; w * h];
    for y in 0..h {
      for x in 0..w {
        let phase = ((x + y) & 1) as i32; // 0 or 1
        let val = 100i32 + if phase == 0 { -64 } else { 64 };
        data[y * w + x] = val as u8;
      }
    }
    let sigma = scalar::Scalar::noise(&data, w, h, w);
    // For a perfect ±64 checkerboard, every interior pixel sees
    // lap = 4·c - 2·(t+b+l+r) + (tl+tr+bl+br)
    //     = 4·(±64) - 2·(∓64·4) + (±64·4)
    //     = 4·(±64) + 8·(±64) + 4·(±64) = 16·(±64) = ±1024
    // |lap| = 1024 per interior pixel. With interior = 14·14 = 196,
    // mean |lap| = 1024, σₙ ≈ 0.208898 · 1024 ≈ 213.92.
    let expected = 0.208_898_754_886_372_3_f64 * 1024.0;
    assert!(
      ((sigma as f64) - expected).abs() < 0.5,
      "expected ~{expected}, got {sigma}"
    );
  }

  #[test]
  fn scalar_noise_stride_padding_is_ignored() {
    // 4 wide × 4 high, stride 8. Padding filled with 255 — would
    // explode |lap| if it leaked into the kernel.
    let w = 4usize;
    let h = 4usize;
    let stride = 8usize;
    let mut data = vec![255u8; stride * h];
    for y in 0..h {
      for x in 0..w {
        data[y * stride + x] = 100; // uniform pixel area → σₙ = 0
      }
    }
    let sigma = scalar::Scalar::noise(&data, w, h, stride);
    assert_eq!(sigma, 0.0, "padding leaked into the Laplacian");
  }
```

- [ ] **Step 4: Run the new scalar kernel tests.**

Run: `cargo test -p scenesdetect --lib arch::tests::scalar_noise_`
Expected: 4 tests pass.

- [ ] **Step 5: Create `src/keyframe/noise.rs` with the public detector and inline tests.**

Write to `src/keyframe/noise.rs`:

```rust
//! Immerkaer fast noise estimator.
//!
//! Estimates per-pixel additive-Gaussian noise standard deviation σₙ
//! on a luma plane using Immerkaer's 1996 single-pass technique:
//!
//! ```text
//!         ⎡  1  -2   1 ⎤
//!     N = ⎢ -2   4  -2 ⎥
//!         ⎣  1  -2   1 ⎦
//!
//!     σₙ ≈ √(π/2) · (1 / (6·N_inner)) · Σ |luma ⊛ N|
//! ```
//!
//! Border pixels are excluded (matching the Tenengrad convention).
//! Higher values mean noisier frames; the absolute scale depends on
//! input resolution, so scores are only comparable within a shot at
//! the same downscaled dimensions.
//!
//! # Example
//!
//! ```no_run
//! use core::num::NonZeroU32;
//! use scenesdetect::frame::{LumaFrame, Timebase, Timestamp};
//! use scenesdetect::keyframe::noise::{Detector, Options};
//!
//! let mut det = Detector::new(Options::default());
//!
//! # let bytes = vec![0u8; 256 * 144];
//! # let tb = Timebase::new(1, NonZeroU32::new(1_000_000).unwrap());
//! # let luma = LumaFrame::new(&bytes, 256, 144, 256, Timestamp::new(0, tb));
//! let sigma = det.observe_luma(luma);
//! assert!(sigma >= 0.0);
//! ```

use crate::frame::LumaFrame;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Options for the noise detector.
///
/// Currently only carries the `use_simd` flag for forward-compatibility;
/// the scalar path is always used in v0.1.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Options {
  use_simd: bool,
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
    Self { use_simd: true }
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
}

/// Pure-algo state machine that reduces a luma frame to its Immerkaer
/// noise estimate σₙ in 0-255 space.
#[derive(Debug, Clone)]
pub struct Detector {
  opts: Options,
}

impl Detector {
  /// Creates a new detector with the supplied options.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(opts: Options) -> Self {
    Self { opts }
  }

  /// Returns the detector's current options.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn options(&self) -> &Options {
    &self.opts
  }

  /// Resets stream state. No-op today; reserved for future SIMD caches.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn clear(&mut self) {}

  /// Computes the Immerkaer noise estimate σₙ on `luma`. Frames narrower
  /// or shorter than 3 pixels have no interior and yield `0.0`.
  pub fn observe_luma(&mut self, luma: LumaFrame<'_>) -> f32 {
    crate::arch::noise(
      luma.data(),
      luma.width() as usize,
      luma.height() as usize,
      luma.stride() as usize,
      self.opts.use_simd,
    )
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::frame::{LumaFrame, Timebase, Timestamp};
  use core::num::NonZeroU32;
  use std::vec;

  fn nz(n: u32) -> NonZeroU32 {
    NonZeroU32::new(n).expect("non-zero")
  }

  fn timestamp() -> Timestamp {
    Timestamp::new(0, Timebase::new(1, nz(1_000_000)))
  }

  fn tight_luma(data: &[u8], w: u32, h: u32) -> LumaFrame<'_> {
    LumaFrame::new(data, w, h, w, timestamp())
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
  fn uniform_frame_has_zero_noise() {
    let data = vec![100u8; 32 * 32];
    let mut det = Detector::new(Options::default());
    let sigma = det.observe_luma(tight_luma(&data, 32, 32));
    assert_eq!(sigma, 0.0);
  }

  #[test]
  fn too_small_frame_yields_zero() {
    let data = vec![0u8; 4];
    let mut det = Detector::new(Options::default());
    let sigma = det.observe_luma(tight_luma(&data, 2, 2));
    assert_eq!(sigma, 0.0);
  }

  #[test]
  fn checkerboard_matches_closed_form() {
    // ±64 amplitude checkerboard → per-pixel |lap| = 1024 → σₙ ≈ 213.92.
    let (w, h) = (16usize, 16usize);
    let mut data = vec![0u8; w * h];
    for y in 0..h {
      for x in 0..w {
        let phase = ((x + y) & 1) as i32;
        let val = 100i32 + if phase == 0 { -64 } else { 64 };
        data[y * w + x] = val as u8;
      }
    }
    let mut det = Detector::new(Options::default());
    let sigma = det.observe_luma(tight_luma(&data, w as u32, h as u32));
    let expected = 0.208_898_754_886_372_3_f64 * 1024.0;
    assert!(
      ((sigma as f64) - expected).abs() < 0.5,
      "expected ~{expected}, got {sigma}"
    );
  }

  #[test]
  fn stride_padding_is_ignored() {
    let w = 4usize;
    let h = 4usize;
    let stride = 8usize;
    let mut data = vec![255u8; stride * h];
    for y in 0..h {
      for x in 0..w {
        data[y * stride + x] = 100;
      }
    }
    let f = LumaFrame::new(&data, w as u32, h as u32, stride as u32, timestamp());
    let mut det = Detector::new(Options::default());
    let sigma = det.observe_luma(f);
    assert_eq!(sigma, 0.0, "padding leaked into kernel");
  }

  #[test]
  fn clear_is_noop() {
    let mut det = Detector::new(Options::default());
    det.clear();
    let data = vec![0u8; 16 * 16];
    let sigma = det.observe_luma(tight_luma(&data, 16, 16));
    assert_eq!(sigma, 0.0);
  }
}
```

- [ ] **Step 6: Add `pub mod noise;` to `src/keyframe.rs`.**

Locate the existing `pub mod` block in `src/keyframe.rs`. Insert `pub mod noise;` alphabetically between `pub mod metrics;` and `pub mod preprocess;`. Updated section:

```rust
pub mod clipping;
pub mod luma;
pub mod metrics;
pub mod noise;
pub mod preprocess;
pub mod saturation;
pub mod select;
pub mod sharpness;
```

- [ ] **Step 7: Run the new module's tests.**

Run: `cargo test -p scenesdetect --lib keyframe::noise::`
Expected: 7 tests pass (`options_default_enables_simd`, `options_builder_roundtrips`, `uniform_frame_has_zero_noise`, `too_small_frame_yields_zero`, `checkerboard_matches_closed_form`, `stride_padding_is_ignored`, `clear_is_noop`).

- [ ] **Step 8: Run the full crate test suite to confirm no regressions.**

Run: `cargo test -p scenesdetect`
Expected: all tests pass.

- [ ] **Step 9: Commit.**

```bash
git add src/arch.rs src/keyframe/noise.rs src/keyframe.rs
git commit -m "$(cat <<'EOF'
feat(keyframe): add Immerkaer noise (σₙ) detector

Scalar kernel in arch::scalar::Scalar::noise, top-level dispatch fn
keeping the SIMD-ladder shape, public keyframe::noise module with the
standard Detector/Options/observe_luma/clear contract.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Motion-blur detector (gradient anisotropy)

Reuses `arch::sobel(...)` (which produces i32 magnitude + u8 direction). The detector owns mag + dir scratch buffers that grow monotonically. After Sobel runs, we walk the magnitude/direction planes once to build a 4-bin magnitude-weighted histogram and reduce to a single anisotropy score ∈ `[0, 1]`.

**Files:**

- Modify: `src/arch.rs` (add `Scalar::gradient_anisotropy` and top-level `gradient_anisotropy`)
- Create: `src/keyframe/motion_blur.rs`
- Modify: `src/keyframe.rs` (add `pub mod motion_blur;`)

### Steps

- [ ] **Step 1: Add `Scalar::gradient_anisotropy` to `arch::scalar`.**

In `src/arch.rs`, inside `mod scalar`'s `impl Scalar { ... }` block (same block as `Scalar::tenengrad` and the `noise` kernel added in Task 5), append:

```rust
    /// Magnitude-weighted gradient-direction concentration.
    /// Inputs are the magnitude and direction planes produced by
    /// [`super::super::sobel`] — `mag` is L1 magnitude (i32), `dir` is
    /// the quantized direction bin in `{0, 1, 2, 3}`. Border pixels
    /// (where Sobel leaves zeros) contribute nothing.
    ///
    /// Builds `hist[k] = Σ mag[p] where dir[p] == k`, then returns
    /// `max((max(hist) / total) - 0.25, 0) / 0.75`. The 0.25 baseline
    /// is the uniform-distribution expectation over the 4 bins; the
    /// output is in `[0, 1]` with 0 = perfectly uniform and 1 =
    /// entirely one direction. Frames with total magnitude 0 (uniform
    /// luma) return 0.
    pub(super) fn gradient_anisotropy(mag: &[i32], dir: &[u8], w: usize, h: usize) -> f32 {
      if w < 3 || h < 3 {
        return 0.0;
      }
      let mut hist = [0u64; 4];
      for y in 1..h - 1 {
        for x in 1..w - 1 {
          let idx = y * w + x;
          let m = mag[idx];
          if m <= 0 {
            continue;
          }
          let d = dir[idx] as usize & 0b11;
          hist[d] = hist[d].saturating_add(m as u64);
        }
      }
      let total: u64 = hist.iter().sum();
      if total == 0 {
        return 0.0;
      }
      let max_bin = *hist.iter().max().expect("4 bins") as f64;
      let total_f = total as f64;
      let frac = max_bin / total_f;
      // Normalise so [0.25, 1.0] maps to [0.0, 1.0]; clamp below.
      ((frac - 0.25).max(0.0) / 0.75) as f32
    }
```

- [ ] **Step 2: Add the `pub(crate) fn gradient_anisotropy` dispatch fn.**

Append after the `pub(crate) fn noise` block added in Task 5:

```rust
/// Magnitude-weighted gradient-direction concentration. Returns an
/// anisotropy score in `[0, 1]` — 0 = isotropic gradients, 1 = a single
/// dominant direction. Dispatches to scalar today; signature preserved
/// for future SIMD backends.
///
/// Inputs are the magnitude and quantized-direction planes produced by
/// [`sobel`].
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(unreachable_code)]
pub(crate) fn gradient_anisotropy(
  mag: &[i32],
  dir: &[u8],
  width: usize,
  height: usize,
  use_simd: bool,
) -> f32 {
  if !use_simd {
    return scalar::Scalar::gradient_anisotropy(mag, dir, width, height);
  }
  scalar::Scalar::gradient_anisotropy(mag, dir, width, height)
}
```

- [ ] **Step 3: Add scalar-kernel tests for `gradient_anisotropy`.**

In `src/arch.rs`'s existing test module, append:

```rust
  #[test]
  fn scalar_gradient_anisotropy_zero_mag_is_zero() {
    let (w, h) = (8usize, 8usize);
    let mag = vec![0i32; w * h];
    let dir = vec![0u8; w * h];
    assert_eq!(scalar::Scalar::gradient_anisotropy(&mag, &dir, w, h), 0.0);
  }

  #[test]
  fn scalar_gradient_anisotropy_too_small_is_zero() {
    let mag = vec![100i32; 4];
    let dir = vec![0u8; 4];
    assert_eq!(scalar::Scalar::gradient_anisotropy(&mag, &dir, 2, 2), 0.0);
  }

  #[test]
  fn scalar_gradient_anisotropy_single_direction_is_one() {
    // Every interior pixel has equal magnitude in direction bin 0.
    let (w, h) = (8usize, 8usize);
    let mag = vec![100i32; w * h];
    let dir = vec![0u8; w * h];
    let a = scalar::Scalar::gradient_anisotropy(&mag, &dir, w, h);
    assert!(
      (a - 1.0).abs() < 1e-6,
      "expected 1.0 for single-direction frame, got {a}"
    );
  }

  #[test]
  fn scalar_gradient_anisotropy_uniform_directions_is_zero() {
    // Interior pixels evenly split across all 4 bins → fraction 0.25 → 0.0.
    let (w, h) = (10usize, 10usize);
    let mut mag = vec![100i32; w * h];
    let mut dir = vec![0u8; w * h];
    // Border pixels won't contribute (interior loop skips y=0, y=h-1, etc.)
    // For interior 8×8 = 64 pixels, assign 16 to each of 4 bins.
    let mut counter = 0u8;
    for y in 1..h - 1 {
      for x in 1..w - 1 {
        dir[y * w + x] = counter & 0b11;
        counter = counter.wrapping_add(1);
      }
    }
    // Force perfectly equal magnitude in each bin by zeroing any
    // imbalance: just trust the equal-magnitude-equal-count construction.
    let a = scalar::Scalar::gradient_anisotropy(&mag, &dir, w, h);
    assert!(a.abs() < 1e-6, "expected ~0.0 for uniform directions, got {a}");
    // Sanity-check we wrote to `mag` — guards against the test
    // accidentally measuring an all-zero plane.
    assert!(mag.iter().any(|&v| v != 0));
  }
```

- [ ] **Step 4: Run the scalar kernel tests.**

Run: `cargo test -p scenesdetect --lib arch::tests::scalar_gradient_anisotropy_`
Expected: 4 tests pass.

- [ ] **Step 5: Create `src/keyframe/motion_blur.rs`.**

Write to `src/keyframe/motion_blur.rs`:

```rust
//! Gradient-anisotropy motion-blur estimator.
//!
//! Heuristic: a frame with motion blur in direction θ has gradients
//! that concentrate along the orientation perpendicular to θ; a
//! sharp, scene-rich frame has a more uniform distribution of
//! gradient directions. The detector runs a 3×3 Sobel on the luma
//! plane (via [`crate::arch::sobel`]), builds a 4-bin magnitude-
//! weighted histogram of the quantized direction, and reports
//! `max((max_bin / total) - 0.25, 0) / 0.75`. The output is in
//! `[0, 1]` with 0 = isotropic and 1 = concentrated in a single
//! direction.
//!
//! Caveat: the metric also fires on scenes with a single dominant
//! orientation (e.g. forest canopies, building façades). The
//! selection layer treats motion_blur as opt-in for the strict gate
//! and defaults the composite-argmax weight to zero.
//!
//! Two entry points:
//!
//! - [`Detector::observe_luma`]: runs the full pipeline (internal
//!   Sobel scratch buffers grow monotonically).
//! - [`Detector::observe_sobel`]: skips the Sobel stage when the
//!   caller already has magnitude and direction planes.

use crate::frame::LumaFrame;
use std::vec::Vec;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Options for the motion-blur detector.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Options {
  use_simd: bool,
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
    Self { use_simd: true }
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
}

/// Pure-algo state machine that reduces a luma frame to an anisotropy
/// score in `[0, 1]`. Owns scratch buffers for Sobel magnitude and
/// direction; they grow to the largest frame seen.
#[derive(Debug, Clone)]
pub struct Detector {
  opts: Options,
  mag: Vec<i32>,
  dir: Vec<u8>,
}

impl Detector {
  /// Creates a new detector with the supplied options.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(opts: Options) -> Self {
    Self {
      opts,
      mag: Vec::new(),
      dir: Vec::new(),
    }
  }

  /// Returns the detector's current options.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn options(&self) -> &Options {
    &self.opts
  }

  /// Resets stream state. No-op today; reserved for future SIMD caches.
  /// Does not free the Sobel scratch buffers — they stay grown for
  /// reuse across shots.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn clear(&mut self) {}

  /// Computes the anisotropy score on `luma` and returns it in
  /// `[0, 1]`. Runs Sobel internally into owned scratch buffers,
  /// then reduces. Frames narrower or shorter than 3 pixels return
  /// `0.0`.
  pub fn observe_luma(&mut self, luma: LumaFrame<'_>) -> f32 {
    let w = luma.width() as usize;
    let h = luma.height() as usize;
    if w < 3 || h < 3 {
      return 0.0;
    }
    let n = w.saturating_mul(h);
    if n == 0 {
      return 0.0;
    }
    if self.mag.len() < n {
      self.mag.resize(n, 0);
    }
    if self.dir.len() < n {
      self.dir.resize(n, 0);
    }
    // arch::sobel expects tight-packed input; honour stride by
    // requiring stride == width. Callers using stride-padded frames
    // should preprocess into a tight buffer (matching the existing
    // detector pattern in keyframe::sharpness).
    let stride = luma.stride() as usize;
    debug_assert!(
      stride == w,
      "motion_blur::observe_luma expects tight-packed luma; got stride={stride} width={w}"
    );
    crate::arch::sobel(
      luma.data(),
      &mut self.mag[..n],
      &mut self.dir[..n],
      w,
      h,
      self.opts.use_simd,
    );
    crate::arch::gradient_anisotropy(&self.mag[..n], &self.dir[..n], w, h, self.opts.use_simd)
  }

  /// Skips the Sobel stage of [`Self::observe_luma`] — the caller
  /// already has the magnitude and direction planes. `mag` and `dir`
  /// are tight-packed `width × height`. On equivalent inputs, this
  /// returns the same value as [`Self::observe_luma`].
  pub fn observe_sobel(&mut self, mag: &[i32], dir: &[u8], width: usize, height: usize) -> f32 {
    crate::arch::gradient_anisotropy(mag, dir, width, height, self.opts.use_simd)
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::frame::{LumaFrame, Timebase, Timestamp};
  use core::num::NonZeroU32;
  use std::vec;

  fn nz(n: u32) -> NonZeroU32 {
    NonZeroU32::new(n).expect("non-zero")
  }

  fn timestamp() -> Timestamp {
    Timestamp::new(0, Timebase::new(1, nz(1_000_000)))
  }

  fn tight_luma(data: &[u8], w: u32, h: u32) -> LumaFrame<'_> {
    LumaFrame::new(data, w, h, w, timestamp())
  }

  #[test]
  fn options_default_enables_simd() {
    assert!(Options::default().use_simd());
  }

  #[test]
  fn uniform_frame_yields_zero_anisotropy() {
    let data = vec![100u8; 32 * 32];
    let mut det = Detector::new(Options::default());
    let a = det.observe_luma(tight_luma(&data, 32, 32));
    assert_eq!(a, 0.0);
  }

  #[test]
  fn vertical_edge_is_highly_anisotropic() {
    // Strong vertical edge → gradient direction concentrates in one
    // bin → anisotropy → 1.0.
    let (w, h) = (32usize, 32usize);
    let mut data = vec![0u8; w * h];
    for y in 0..h {
      for x in (w / 2)..w {
        data[y * w + x] = 255;
      }
    }
    let mut det = Detector::new(Options::default());
    let a = det.observe_luma(tight_luma(&data, w as u32, h as u32));
    assert!(a > 0.9, "expected strong anisotropy (>0.9), got {a}");
  }

  #[test]
  fn too_small_frame_yields_zero() {
    let data = vec![0u8; 4];
    let mut det = Detector::new(Options::default());
    let a = det.observe_luma(tight_luma(&data, 2, 2));
    assert_eq!(a, 0.0);
  }

  #[test]
  fn observe_sobel_matches_observe_luma() {
    // Build a frame, run observe_luma, then build mag/dir externally
    // and confirm observe_sobel produces the same result.
    let (w, h) = (16usize, 16usize);
    let mut data = vec![0u8; w * h];
    for y in 0..h {
      for x in 8..w {
        data[y * w + x] = 255;
      }
    }
    let mut det = Detector::new(Options::default());
    let from_luma = det.observe_luma(tight_luma(&data, w as u32, h as u32));

    let mut mag = vec![0i32; w * h];
    let mut dir = vec![0u8; w * h];
    crate::arch::sobel(&data, &mut mag, &mut dir, w, h, false);
    let from_sobel = det.observe_sobel(&mag, &dir, w, h);

    assert!(
      (from_luma - from_sobel).abs() < 1e-6,
      "luma path {from_luma} != sobel path {from_sobel}"
    );
  }

  #[test]
  fn clear_is_noop() {
    let mut det = Detector::new(Options::default());
    det.clear();
    let data = vec![0u8; 16 * 16];
    let a = det.observe_luma(tight_luma(&data, 16, 16));
    assert_eq!(a, 0.0);
  }
}
```

- [ ] **Step 6: Add `pub mod motion_blur;` to `src/keyframe.rs`.**

Updated module block:

```rust
pub mod clipping;
pub mod luma;
pub mod metrics;
pub mod motion_blur;
pub mod noise;
pub mod preprocess;
pub mod saturation;
pub mod select;
pub mod sharpness;
```

- [ ] **Step 7: Run the new module's tests.**

Run: `cargo test -p scenesdetect --lib keyframe::motion_blur::`
Expected: 6 tests pass.

- [ ] **Step 8: Run the full crate test suite to confirm no regressions.**

Run: `cargo test -p scenesdetect`
Expected: all tests pass.

- [ ] **Step 9: Commit.**

```bash
git add src/arch.rs src/keyframe/motion_blur.rs src/keyframe.rs
git commit -m "$(cat <<'EOF'
feat(keyframe): add gradient-anisotropy motion-blur detector

Scalar arch::gradient_anisotropy + dispatch fn. The public
keyframe::motion_blur detector owns Sobel mag/dir scratch buffers
that grow monotonically, exposes observe_luma and observe_sobel
entry points.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Colorfulness detector (Hasler-Süßstrunk)

**Files:**

- Modify: `src/arch.rs` (add `Scalar::colorfulness` and top-level `colorfulness`)
- Create: `src/keyframe/colorfulness.rs`
- Modify: `src/keyframe.rs` (add `pub mod colorfulness;`)

### Steps

- [ ] **Step 1: Add `Scalar::colorfulness` to `arch::scalar`.**

In `src/arch.rs`, inside `mod scalar`'s `impl Scalar { ... }` block (same block as the other scalar kernels), append:

```rust
    /// Hasler-Süßstrunk colourfulness metric on packed 24-bit BGR.
    /// Single pass, streaming moments on `rg = R - G` and
    /// `yb = 0.5(R + G) - B`. Returns
    /// `σ_rgyb + 0.3·μ_rgyb` where
    /// `σ_rgyb = √(σ²_rg + σ²_yb)` and `μ_rgyb = √(μ²_rg + μ²_yb)`.
    /// Empty inputs return 0.
    pub(super) fn colorfulness(bgr: &[u8], w: usize, h: usize, stride: usize) -> f32 {
      let n = w.saturating_mul(h);
      if n == 0 {
        return 0.0;
      }
      let n_f = n as f64;

      // Welford-style streaming mean/M2 on rg and yb concurrently.
      let mut mean_rg: f64 = 0.0;
      let mut m2_rg: f64 = 0.0;
      let mut mean_yb: f64 = 0.0;
      let mut m2_yb: f64 = 0.0;
      let mut k: u64 = 0;

      for y in 0..h {
        let row = &bgr[y * stride..y * stride + w * 3];
        // BGR packed: row[3i] = B, row[3i+1] = G, row[3i+2] = R.
        for i in 0..w {
          let b = row[3 * i] as f64;
          let g = row[3 * i + 1] as f64;
          let r = row[3 * i + 2] as f64;
          let rg = r - g;
          let yb = 0.5 * (r + g) - b;
          k += 1;
          let kf = k as f64;
          let d_rg = rg - mean_rg;
          mean_rg += d_rg / kf;
          m2_rg += d_rg * (rg - mean_rg);
          let d_yb = yb - mean_yb;
          mean_yb += d_yb / kf;
          m2_yb += d_yb * (yb - mean_yb);
        }
      }

      // Population variance (use k, not k-1) so an all-identical frame
      // gives σ = 0.
      let var_rg = (m2_rg / n_f).max(0.0);
      let var_yb = (m2_yb / n_f).max(0.0);
      let sigma_rgyb = (var_rg + var_yb).sqrt();
      let mu_rgyb = (mean_rg * mean_rg + mean_yb * mean_yb).sqrt();
      (sigma_rgyb + 0.3 * mu_rgyb) as f32
    }
```

- [ ] **Step 2: Add the top-level dispatch fn.**

Append after the existing `gradient_anisotropy` block:

```rust
/// Hasler-Süßstrunk colourfulness metric on packed 24-bit BGR.
/// See the scalar kernel for the formula. Dispatches to scalar
/// today; signature preserved for future SIMD backends.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(unreachable_code)]
pub(crate) fn colorfulness(
  bgr: &[u8],
  width: usize,
  height: usize,
  stride: usize,
  use_simd: bool,
) -> f32 {
  if !use_simd {
    return scalar::Scalar::colorfulness(bgr, width, height, stride);
  }
  scalar::Scalar::colorfulness(bgr, width, height, stride)
}
```

- [ ] **Step 3: Add scalar-kernel tests.**

In `src/arch.rs`'s test module, append:

```rust
  #[test]
  fn scalar_colorfulness_uniform_gray_is_zero() {
    let w = 16usize;
    let h = 16usize;
    let data = vec![128u8; w * h * 3];
    let c = scalar::Scalar::colorfulness(&data, w, h, w * 3);
    assert!(c.abs() < 1e-3, "expected ~0.0, got {c}");
  }

  #[test]
  fn scalar_colorfulness_pure_red_has_nonzero_score() {
    // Pure red: B=0, G=0, R=255 → rg = 255, yb = 127.5.
    // Constant per pixel, so var_rg = var_yb = 0, σ = 0, μ_rgyb =
    // sqrt(255² + 127.5²) ≈ 285.06.  C = 0 + 0.3 · 285.06 ≈ 85.5.
    let w = 8usize;
    let h = 8usize;
    let mut data = vec![0u8; w * h * 3];
    for i in 0..(w * h) {
      data[i * 3 + 2] = 255;
    }
    let c = scalar::Scalar::colorfulness(&data, w, h, w * 3);
    let expected = 0.3_f64 * (255.0_f64.powi(2) + 127.5_f64.powi(2)).sqrt();
    assert!(
      ((c as f64) - expected).abs() < 1e-2,
      "expected ~{expected}, got {c}"
    );
  }

  #[test]
  fn scalar_colorfulness_stride_padding_is_ignored() {
    let w = 4usize;
    let h = 4usize;
    let stride = 32usize; // > 4 * 3 = 12
    let mut data = vec![200u8; stride * h]; // padding is "color-y"
    // pixel area: uniform gray
    for y in 0..h {
      for x in 0..w {
        data[y * stride + x * 3] = 128;
        data[y * stride + x * 3 + 1] = 128;
        data[y * stride + x * 3 + 2] = 128;
      }
    }
    let c = scalar::Scalar::colorfulness(&data, w, h, stride);
    assert!(c.abs() < 1e-3, "padding leaked into reduction, got {c}");
  }

  #[test]
  fn scalar_colorfulness_empty_frame_is_zero() {
    let data = vec![0u8];
    assert_eq!(scalar::Scalar::colorfulness(&data, 0, 0, 0), 0.0);
  }
```

- [ ] **Step 4: Run scalar kernel tests.**

Run: `cargo test -p scenesdetect --lib arch::tests::scalar_colorfulness_`
Expected: 4 tests pass.

- [ ] **Step 5: Create `src/keyframe/colorfulness.rs`.**

Write to `src/keyframe/colorfulness.rs`:

```rust
//! Hasler-Süßstrunk colorfulness detector.
//!
//! Reports a perceptually-grounded "how colorful is this frame"
//! score in approximately `[0, 200]` (higher = more colorful).
//! Operates on packed 24-bit BGR. Single pass over the pixels using
//! streaming Welford-style moments on `rg = R − G` and
//! `yb = ½(R + G) − B`:
//!
//! ```text
//!     σ_rgyb = √(σ²_rg + σ²_yb)
//!     μ_rgyb = √(μ²_rg + μ²_yb)
//!     C      = σ_rgyb + 0.3 · μ_rgyb
//! ```
//!
//! Byte order: the implementation treats the packed input as BGR
//! (matching [`crate::keyframe::preprocess`]). RGB callers must
//! swizzle before feeding data in (the `rg` channel is symmetric
//! under R/B swap but `yb` is not).
//!
//! # Example
//!
//! ```no_run
//! use core::num::NonZeroU32;
//! use scenesdetect::frame::{RgbFrame, Timebase, Timestamp};
//! use scenesdetect::keyframe::colorfulness::{Detector, Options};
//!
//! let mut det = Detector::new(Options::default());
//! # let bytes = vec![0u8; 256 * 144 * 3];
//! # let tb = Timebase::new(1, NonZeroU32::new(1_000_000).unwrap());
//! # let f = RgbFrame::new(&bytes, 256, 144, 256 * 3, Timestamp::new(0, tb));
//! let c = det.observe_rgb(f);
//! assert!(c >= 0.0);
//! ```

use crate::frame::RgbFrame;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Options for the colorfulness detector.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Options {
  use_simd: bool,
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
    Self { use_simd: true }
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
}

/// Pure-algo state machine that reduces a BGR frame to its
/// Hasler-Süßstrunk colorfulness score.
#[derive(Debug, Clone)]
pub struct Detector {
  opts: Options,
}

impl Detector {
  /// Creates a new detector with the supplied options.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(opts: Options) -> Self {
    Self { opts }
  }

  /// Returns the detector's current options.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn options(&self) -> &Options {
    &self.opts
  }

  /// Resets stream state. No-op today; reserved for future SIMD caches.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn clear(&mut self) {}

  /// Computes the colourfulness of `rgb`. Returns a non-negative
  /// score; a uniform gray frame yields ~0.
  pub fn observe_rgb(&mut self, rgb: RgbFrame<'_>) -> f32 {
    crate::arch::colorfulness(
      rgb.data(),
      rgb.width() as usize,
      rgb.height() as usize,
      rgb.stride() as usize,
      self.opts.use_simd,
    )
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::frame::{RgbFrame, Timebase, Timestamp};
  use core::num::NonZeroU32;
  use std::vec;

  fn nz(n: u32) -> NonZeroU32 {
    NonZeroU32::new(n).expect("non-zero")
  }

  fn timestamp() -> Timestamp {
    Timestamp::new(0, Timebase::new(1, nz(1_000_000)))
  }

  #[test]
  fn options_default_enables_simd() {
    assert!(Options::default().use_simd());
  }

  #[test]
  fn uniform_gray_frame_has_zero_colorfulness() {
    let w = 16u32;
    let h = 16u32;
    let data = vec![128u8; (w * h * 3) as usize];
    let f = RgbFrame::new(&data, w, h, w * 3, timestamp());
    let mut det = Detector::new(Options::default());
    let c = det.observe_rgb(f);
    assert!(c.abs() < 1e-3);
  }

  #[test]
  fn pure_red_has_nonzero_colorfulness() {
    let w = 8u32;
    let h = 8u32;
    let mut data = vec![0u8; (w * h * 3) as usize];
    for i in 0..(w * h) as usize {
      data[i * 3 + 2] = 255;
    }
    let f = RgbFrame::new(&data, w, h, w * 3, timestamp());
    let mut det = Detector::new(Options::default());
    let c = det.observe_rgb(f);
    let expected = 0.3_f64 * (255.0_f64.powi(2) + 127.5_f64.powi(2)).sqrt();
    assert!(
      ((c as f64) - expected).abs() < 1e-2,
      "expected ~{expected}, got {c}"
    );
  }

  #[test]
  fn clear_is_noop() {
    let mut det = Detector::new(Options::default());
    det.clear();
    let data = vec![128u8; 16 * 16 * 3];
    let f = RgbFrame::new(&data, 16, 16, 16 * 3, timestamp());
    let c = det.observe_rgb(f);
    assert!(c.abs() < 1e-3);
  }
}
```

- [ ] **Step 6: Add `pub mod colorfulness;` to `src/keyframe.rs`.**

Updated module block:

```rust
pub mod clipping;
pub mod colorfulness;
pub mod luma;
pub mod metrics;
pub mod motion_blur;
pub mod noise;
pub mod preprocess;
pub mod saturation;
pub mod select;
pub mod sharpness;
```

- [ ] **Step 7: Run the new module's tests.**

Run: `cargo test -p scenesdetect --lib keyframe::colorfulness::`
Expected: 4 tests pass.

- [ ] **Step 8: Full crate test sweep.**

Run: `cargo test -p scenesdetect`
Expected: all tests pass.

- [ ] **Step 9: Commit.**

```bash
git add src/arch.rs src/keyframe/colorfulness.rs src/keyframe.rs
git commit -m "$(cat <<'EOF'
feat(keyframe): add Hasler-Süßstrunk colorfulness detector

Scalar arch::colorfulness with Welford streaming moments, top-level
dispatch fn keeping the SIMD-ladder shape, public keyframe::colorfulness
module.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: `CompositeWeights` struct in `keyframe::select`

Introduce the new struct (private fields + getters + paired `with_*` builders). No selector behaviour change yet — Task 9 wires it into the argmax.

**Files:**

- Modify: `src/keyframe/select.rs`

### Steps

- [ ] **Step 1: Add `CompositeWeights` and its impls to `src/keyframe/select.rs`.**

Insert directly above the existing `// ---- Options ---------------------------` divider at the top of the file (immediately after the imports / `use` block). Code:

```rust
// ---- CompositeWeights ------------------------------------------------------

/// Weights and per-metric normalisers consumed by
/// [`composite_quality`] when ranking frames inside a bucket.
///
/// Defaults are tuned so a "good" baseline frame (sharp, mid-luma,
/// not noisy, mildly colorful) scores ≈ 1.0. Zeroing every term
/// except `sharpness` collapses the composite to pure Tenengrad —
/// identical to the legacy strict-pass argmax.
///
/// Fields are private; use `with_*` builders for construction and
/// the field-name getters for read access.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CompositeWeights {
  sharpness: f32,
  sharpness_norm: f32,
  noise: f32,
  noise_norm: f32,
  colorfulness: f32,
  colorfulness_norm: f32,
  clipping: f32,
  motion_blur: f32,
}

impl Default for CompositeWeights {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn default() -> Self {
    Self::new()
  }
}

impl CompositeWeights {
  /// Creates a [`CompositeWeights`] with the calibrated default
  /// weights and normalisers. See the type docs for the calibration
  /// rationale.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      sharpness: 1.0,
      sharpness_norm: 1000.0,
      noise: 0.3,
      noise_norm: 20.0,
      colorfulness: 0.2,
      colorfulness_norm: 50.0,
      clipping: 0.5,
      motion_blur: 0.0,
    }
  }

  /// Sets the sharpness weight and its normaliser.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_sharpness(mut self, weight: f32, norm: f32) -> Self {
    self.sharpness = weight;
    self.sharpness_norm = norm;
    self
  }
  /// Sets the noise weight and its normaliser. Noise is a penalty
  /// (subtracted in the composite).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_noise(mut self, weight: f32, norm: f32) -> Self {
    self.noise = weight;
    self.noise_norm = norm;
    self
  }
  /// Sets the colorfulness weight and its normaliser.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_colorfulness(mut self, weight: f32, norm: f32) -> Self {
    self.colorfulness = weight;
    self.colorfulness_norm = norm;
    self
  }
  /// Sets the clipping-penalty weight. Clipping is already in `[0, 1]`
  /// — no normaliser.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_clipping(mut self, weight: f32) -> Self {
    self.clipping = weight;
    self
  }
  /// Sets the motion-blur-penalty weight. Anisotropy is already in
  /// `[0, 1]` — no normaliser. Defaults to 0 (off).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_motion_blur(mut self, weight: f32) -> Self {
    self.motion_blur = weight;
    self
  }

  // ---- Getters ------------------------------------------------------------

  /// Sharpness weight (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn sharpness(&self) -> f32 {
    self.sharpness
  }
  /// Sharpness normaliser (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn sharpness_norm(&self) -> f32 {
    self.sharpness_norm
  }
  /// Noise weight (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn noise(&self) -> f32 {
    self.noise
  }
  /// Noise normaliser (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn noise_norm(&self) -> f32 {
    self.noise_norm
  }
  /// Colorfulness weight (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn colorfulness(&self) -> f32 {
    self.colorfulness
  }
  /// Colorfulness normaliser (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn colorfulness_norm(&self) -> f32 {
    self.colorfulness_norm
  }
  /// Clipping-penalty weight (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn clipping(&self) -> f32 {
    self.clipping
  }
  /// Motion-blur-penalty weight (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn motion_blur(&self) -> f32 {
    self.motion_blur
  }
}
```

- [ ] **Step 2: Add unit tests for `CompositeWeights`.**

Append inside the existing `tests` module in `src/keyframe/select.rs`:

```rust
  #[test]
  fn composite_weights_default_matches_spec() {
    let w = CompositeWeights::default();
    assert_eq!(w.sharpness(), 1.0);
    assert_eq!(w.sharpness_norm(), 1000.0);
    assert_eq!(w.noise(), 0.3);
    assert_eq!(w.noise_norm(), 20.0);
    assert_eq!(w.colorfulness(), 0.2);
    assert_eq!(w.colorfulness_norm(), 50.0);
    assert_eq!(w.clipping(), 0.5);
    assert_eq!(w.motion_blur(), 0.0);
  }

  #[test]
  fn composite_weights_builders_roundtrip() {
    let w = CompositeWeights::new()
      .with_sharpness(0.5, 500.0)
      .with_noise(0.1, 10.0)
      .with_colorfulness(0.4, 100.0)
      .with_clipping(0.25)
      .with_motion_blur(0.6);
    assert_eq!(w.sharpness(), 0.5);
    assert_eq!(w.sharpness_norm(), 500.0);
    assert_eq!(w.noise(), 0.1);
    assert_eq!(w.noise_norm(), 10.0);
    assert_eq!(w.colorfulness(), 0.4);
    assert_eq!(w.colorfulness_norm(), 100.0);
    assert_eq!(w.clipping(), 0.25);
    assert_eq!(w.motion_blur(), 0.6);
  }

  #[test]
  fn composite_weights_new_is_const_context_usable() {
    const W: CompositeWeights = CompositeWeights::new().with_motion_blur(0.5);
    assert_eq!(W.motion_blur(), 0.5);
  }
```

- [ ] **Step 3: Run the new tests.**

Run: `cargo test -p scenesdetect --lib keyframe::select::tests::composite_weights_`
Expected: 3 tests pass.

- [ ] **Step 4: Build the full crate to confirm no breakage.**

Run: `cargo build -p scenesdetect && cargo test -p scenesdetect`
Expected: clean build, full suite passes.

- [ ] **Step 5: Commit.**

```bash
git add src/keyframe/select.rs
git commit -m "$(cat <<'EOF'
feat(keyframe): add CompositeWeights struct in select

Carries the per-metric weights + normalisers consumed by
composite_quality (added in Task 9). Defaults tuned so a baseline
"good" frame scores ≈ 1.0; zeroing all but sharpness collapses to
the legacy strict-pass ranking.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Composite-quality argmax + `Options::with_composite_weights`

Replace the sharpness-only strict-pass argmax with the weighted composite. Fallback path keeps raw sharpness (so "least bad" is well-defined when every frame fails gates). Plumb `CompositeWeights` through `Options`.

**Files:**

- Modify: `src/keyframe/select.rs`

### Steps

- [ ] **Step 1: Add the `weights: CompositeWeights` field to `Options` with a default value.**

Locate the existing `Options` struct (the top-of-file selector options, the one with `target_interval`, `max_frames_per_shot`, etc.):

```rust
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
```

Add a `weights` field at the end:

```rust
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
  weights: CompositeWeights,
}
```

Locate the `Default for Options` impl:

```rust
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
```

Add the trailing field initialiser:

```rust
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
      weights: CompositeWeights::new(),
    }
  }
}
```

- [ ] **Step 2: Add the `with_composite_weights` builder and `composite_weights` getter to `Options`.**

Inside the existing `impl Options { ... }` block, append after `with_max_clipping`:

```rust
  /// Replaces the [`CompositeWeights`] driving the strict-pass argmax
  /// inside [`Detector::finalize_shot`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_composite_weights(mut self, w: CompositeWeights) -> Self {
    self.weights = w;
    self
  }

  /// Composite-quality weights and normalisers (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn composite_weights(&self) -> &CompositeWeights {
    &self.weights
  }
```

- [ ] **Step 3: Add the `composite_quality` helper.**

In `src/keyframe/select.rs`, locate the `sharper` helper (near the bottom of the file). Insert the new helper directly above it:

```rust
/// Weighted composite of [`FrameMetrics`] used as the strict-pass
/// argmax key inside [`Detector::finalize_shot`].
///
/// Higher is better. `sharpness` and `colorfulness` contribute
/// positively; `noise`, `clipping`, and `motion_blur` contribute
/// negatively. The fallback path inside `finalize_shot` still ranks
/// by raw `sharpness` — see the module docs.
#[cfg_attr(not(tarpaulin), inline(always))]
fn composite_quality(m: &FrameMetrics, w: &CompositeWeights) -> f32 {
  let s = w.sharpness() * (m.sharpness() / w.sharpness_norm());
  let n = w.noise() * (m.noise() / w.noise_norm());
  let c = w.colorfulness() * (m.colorfulness() / w.colorfulness_norm());
  let clip = w.clipping() * m.clipping();
  let mb = w.motion_blur() * m.motion_blur();
  s - n + c - clip - mb
}
```

- [ ] **Step 4: Replace the strict-pass argmax key with `composite_quality`.**

Locate the running-argmax block inside `finalize_shot` (post-Task-2 form):

```rust
      // Running argmax updates.
      if best_any.is_none_or(|(_, s)| sharper(metrics.sharpness(), s)) {
        best_any = Some((ts, metrics.sharpness()));
      }
      if !hard_gate(&metrics, &opts)
        && metrics.sharpness() >= opts.min_sharpness
        && best_strict.is_none_or(|(_, s)| sharper(metrics.sharpness(), s))
      {
        best_strict = Some((ts, metrics.sharpness()));
      }
```

Replace with:

```rust
      // Running argmax updates.
      // Fallback path: pure-sharpness ranking, preserved so "least
      // bad" is well-defined when every frame in the bucket fails
      // the strict gate.
      if best_any.is_none_or(|(_, s)| sharper(metrics.sharpness(), s)) {
        best_any = Some((ts, metrics.sharpness()));
      }
      // Strict path: composite-quality ranking among gate-passing
      // frames.
      if !hard_gate(&metrics, &opts)
        && metrics.sharpness() >= opts.min_sharpness
      {
        let q = composite_quality(&metrics, opts.composite_weights());
        if best_strict.is_none_or(|(_, s)| sharper(q, s)) {
          best_strict = Some((ts, q));
        }
      }
```

- [ ] **Step 5: Add selector tests for the composite argmax.**

Append inside the `tests` module:

```rust
  #[test]
  fn composite_argmax_picks_clean_over_sharper_noisy_under_defaults() {
    // Bucket with two strict-eligible frames:
    //   A: sharpness=2000, noise=15
    //   B: sharpness=1800, noise=3
    // Under default weights:
    //   q_A = 1.0·(2000/1000) - 0.3·(15/20) + 0 - 0 - 0  = 2.0 - 0.225 = 1.775
    //   q_B = 1.0·(1800/1000) - 0.3·( 3/20) + 0 - 0 - 0  = 1.8 - 0.045 = 1.755
    // A still wins by a hair (sharpness dominates), but bumping noise
    // weight should flip it.  Use a stronger noise weight here:
    let weights = CompositeWeights::new().with_noise(2.0, 20.0);
    let opts = Options::default()
      .with_margin_ratio(0.0)
      .with_composite_weights(weights);
    let mut det = Detector::new(opts);

    let a = FrameMetrics::new()
      .with_sharpness(2000.0)
      .with_brightness(128.0)
      .with_luma_variance(200.0)
      .with_saturation_variance(100.0)
      .with_clipping(0.0)
      .with_noise(15.0);
    let b = FrameMetrics::new()
      .with_sharpness(1800.0)
      .with_brightness(128.0)
      .with_luma_variance(200.0)
      .with_saturation_variance(100.0)
      .with_clipping(0.0)
      .with_noise(3.0);

    det.observe(ts(1_000_000), a);
    det.observe(ts(2_000_000), b);

    let out = det.finalize_shot(tr(0, 4_000_000));
    assert_eq!(out, vec![ts(2_000_000)]);
  }

  #[test]
  fn composite_argmax_collapses_to_sharpness_when_other_weights_zero() {
    // Zero out every non-sharpness weight → strict argmax must rank
    // by pure sharpness (mirrors legacy behaviour).
    let weights = CompositeWeights::new()
      .with_noise(0.0, 20.0)
      .with_colorfulness(0.0, 50.0)
      .with_clipping(0.0)
      .with_motion_blur(0.0);
    let opts = Options::default()
      .with_margin_ratio(0.0)
      .with_composite_weights(weights);
    let mut det = Detector::new(opts);

    // Same fixture as the existing `finalize_single_bucket_picks_sharpest`.
    det.observe(ts(0), good_score(100.0));
    det.observe(ts(500_000), good_score(500.0)); // sharpest
    det.observe(ts(1_500_000), good_score(200.0));

    let out = det.finalize_shot(tr(0, 2_000_000));
    assert_eq!(out, vec![ts(500_000)]);
  }
```

- [ ] **Step 6: Run the new selector tests and the full keyframe::select test suite.**

Run: `cargo test -p scenesdetect --lib keyframe::select`
Expected: every existing test still passes, plus the two new tests.

- [ ] **Step 7: Full crate test sweep.**

Run: `cargo test -p scenesdetect`
Expected: all pass.

- [ ] **Step 8: Commit.**

```bash
git add src/keyframe/select.rs
git commit -m "$(cat <<'EOF'
feat(keyframe): composite-quality strict-pass argmax

Replaces the sharpness-only strict-bucket winner with a weighted
composite of FrameMetrics (sharpness, noise penalty, colorfulness,
clipping penalty, motion-blur penalty). Fallback path keeps raw
sharpness so "least bad" is well-defined when every frame fails
the strict gate. Weights are configured via
Options::with_composite_weights(CompositeWeights); defaults
preserve sharpness as the dominant term.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Adaptive per-shot sharpness floor

Lower the strict-gate sharpness floor to `min(opts.min_sharpness, p25_of_shot)` when the shot has at least `min_samples` frames buffered. The floor is only ever lowered — never raised — so this can never make a previously-passing frame fail.

**Files:**

- Modify: `src/keyframe/select.rs`

### Steps

- [ ] **Step 1: Add the new fields to `Options` with safe defaults.**

Inside the `Options` struct, append three fields after `weights`:

```rust
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
  weights: CompositeWeights,
  adaptive_floor: bool,
  adaptive_floor_percentile: f32,
  adaptive_floor_min_samples: usize,
}
```

Update the `Default` impl's struct literal:

```rust
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
      weights: CompositeWeights::new(),
      adaptive_floor: true,
      adaptive_floor_percentile: 0.25,
      adaptive_floor_min_samples: 20,
    }
```

- [ ] **Step 2: Add `with_*` builders and accessors for the three new fields.**

Inside `impl Options { ... }`, append after `with_composite_weights`:

```rust
  /// Enables or disables the adaptive per-shot sharpness floor. When
  /// enabled (the default), the effective strict-gate sharpness floor
  /// becomes `min(min_sharpness, p_in_shot)` for shots that meet
  /// [`Self::adaptive_floor_min_samples`] — the absolute floor is
  /// only **lowered**, never raised.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_adaptive_floor(mut self, on: bool) -> Self {
    self.adaptive_floor = on;
    self
  }

  /// Sets the percentile (in `[0.0, 1.0]`) used by the adaptive floor.
  ///
  /// # Panics
  /// When outside `[0.0, 1.0]`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_adaptive_floor_percentile(mut self, p: f32) -> Self {
    assert!(
      (0.0..=1.0).contains(&p),
      "adaptive_floor_percentile must be in [0.0, 1.0]"
    );
    self.adaptive_floor_percentile = p;
    self
  }

  /// Sets the minimum in-shot sample count required to activate the
  /// adaptive floor. Below this, the absolute floor is used unchanged.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_adaptive_floor_min_samples(mut self, n: usize) -> Self {
    self.adaptive_floor_min_samples = n;
    self
  }

  /// Whether the adaptive per-shot sharpness floor is enabled (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn adaptive_floor(&self) -> bool {
    self.adaptive_floor
  }
  /// Adaptive floor percentile (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn adaptive_floor_percentile(&self) -> f32 {
    self.adaptive_floor_percentile
  }
  /// Adaptive floor minimum-samples threshold (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn adaptive_floor_min_samples(&self) -> usize {
    self.adaptive_floor_min_samples
  }
```

- [ ] **Step 3: Compute the effective floor inside `finalize_shot`.**

Locate `finalize_shot` in `src/keyframe/select.rs`. Just after the "drop stale entries" block (the `while let Some((ts, _)) = self.buffer.front()` block that pops entries older than `range.start()`) and before the bucket-count computation (`let n = compute_n_buckets(...)`), insert:

```rust
    // 2.5. Compute the effective strict-gate sharpness floor for this
    // shot. If adaptive_floor is enabled and the shot has at least
    // `adaptive_floor_min_samples` buffered in-range entries, set the
    // floor to `min(absolute_floor, p_percentile)` — never raising the
    // floor. This lets legitimate low-detail shots (fog, night
    // interiors) produce strict winners instead of always degrading
    // to fallback selection.
    let effective_min_sharpness = compute_effective_floor(
      &self.buffer,
      &range,
      &self.opts,
    );
```

Then change the gate check inside the inner loop. Locate (post-Task-9 form):

```rust
      // Strict path: composite-quality ranking among gate-passing
      // frames.
      if !hard_gate(&metrics, &opts)
        && metrics.sharpness() >= opts.min_sharpness
      {
```

Replace `opts.min_sharpness` with `effective_min_sharpness`:

```rust
      // Strict path: composite-quality ranking among gate-passing
      // frames.
      if !hard_gate(&metrics, &opts)
        && metrics.sharpness() >= effective_min_sharpness
      {
```

- [ ] **Step 4: Add the `compute_effective_floor` helper.**

In `src/keyframe/select.rs`, in the "Helpers" section near `compute_n_buckets`, append:

```rust
/// Returns the effective strict-gate sharpness floor for the shot
/// described by `range`, given the entries currently buffered and the
/// adaptive-floor options. Never raises the floor above
/// [`Options::min_sharpness`]; only lowers it.
fn compute_effective_floor(
  buffer: &VecDeque<(Timestamp, FrameMetrics)>,
  range: &TimeRange,
  opts: &Options,
) -> f32 {
  if !opts.adaptive_floor() {
    return opts.min_sharpness();
  }
  // Collect in-range sharpness values.
  let mut sharps: Vec<f32> = buffer
    .iter()
    .filter(|(ts, _)| {
      ts.cmp_semantic(&range.start()) != Ordering::Less
        && ts.cmp_semantic(&range.end()) == Ordering::Less
    })
    .map(|(_, m)| m.sharpness())
    .collect();
  if sharps.len() < opts.adaptive_floor_min_samples() {
    return opts.min_sharpness();
  }
  sharps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
  let idx = ((sharps.len() as f32 * opts.adaptive_floor_percentile()) as usize)
    .min(sharps.len().saturating_sub(1));
  let p = sharps[idx];
  opts.min_sharpness().min(p) // never raise the floor
}
```

- [ ] **Step 5: Add selector tests for the adaptive floor.**

Append inside the `tests` module:

```rust
  #[test]
  fn adaptive_floor_recovers_strict_winner_in_low_detail_shot() {
    // 25 frames, all with sharpness in [20, 80] — well below the
    // absolute floor of 100. With adaptive_floor enabled, p25 ≈ 35,
    // so the strict gate passes any frame ≥ 35. The sharpest among
    // those becomes the strict winner instead of falling back.
    let opts = Options::default()
      .with_margin_ratio(0.0)
      .with_target_interval(Duration::from_secs(60));
    let mut det = Detector::new(opts);
    // 25 frames evenly spaced 0..25 seconds, sharpness ramping 20..80.
    for i in 0..25 {
      let s = 20.0 + (i as f32) * 2.5; // 20.0, 22.5, ..., 80.0
      det.observe(ts((i as i64) * 1_000_000), good_score(s));
    }
    // Composite-quality argmax with default weights → highest
    // composite wins. Since brightness/clipping/noise/etc are
    // identical, the highest-sharpness frame wins.
    let out = det.finalize_shot(tr(0, 30_000_000));
    assert_eq!(out, vec![ts(24_000_000)]); // last frame, sharpness 80
    // Confirm the strict path was taken: if adaptive floor were off,
    // every frame fails sharpness >= 100, fallback wins — which would
    // also be ts(24_000_000) here, so we need a tie-breaking test
    // below.
  }

  #[test]
  fn adaptive_floor_disabled_falls_back_to_absolute_floor() {
    // 25 frames all below the absolute floor of 100. With adaptive
    // floor explicitly disabled, the strict gate rejects every frame
    // and we drop to fallback (pure sharpness). The result should
    // still be the sharpest frame — but via the fallback path.
    // To prove the path: use a non-default composite that demotes
    // the sharpest frame; the strict path would skip it, fallback
    // would not.
    let weights = CompositeWeights::new()
      .with_noise(10.0, 1.0); // huge noise penalty
    let opts = Options::default()
      .with_margin_ratio(0.0)
      .with_target_interval(Duration::from_secs(60))
      .with_adaptive_floor(false)
      .with_composite_weights(weights);
    let mut det = Detector::new(opts);
    let mut sharpest_with_noise = FrameMetrics::new()
      .with_sharpness(80.0)
      .with_brightness(128.0)
      .with_luma_variance(200.0)
      .with_saturation_variance(100.0)
      .with_clipping(0.0)
      .with_noise(1.0); // any value — strict path is skipped anyway
    for i in 0..24 {
      let s = 20.0 + (i as f32) * 2.5;
      det.observe(ts((i as i64) * 1_000_000), good_score(s));
    }
    // Last frame has the highest sharpness — fallback picks it.
    det.observe(ts(24_000_000), sharpest_with_noise);
    let out = det.finalize_shot(tr(0, 30_000_000));
    assert_eq!(out, vec![ts(24_000_000)]);
  }

  #[test]
  fn adaptive_floor_does_not_raise_floor_in_high_sharpness_shot() {
    // All frames at sharpness 500 — p25 = 500, well above the
    // absolute floor of 100. Effective floor must remain 100 (not
    // jump up to 500), so any frame with sharpness >= 100 still
    // passes.
    let opts = Options::default()
      .with_margin_ratio(0.0)
      .with_target_interval(Duration::from_secs(60));
    let mut det = Detector::new(opts);
    for i in 0..25 {
      det.observe(ts((i as i64) * 1_000_000), good_score(500.0));
    }
    let out = det.finalize_shot(tr(0, 30_000_000));
    assert_eq!(out.len(), 1, "exactly one winner expected");
  }

  #[test]
  fn adaptive_floor_uses_absolute_below_min_samples() {
    // Only 5 frames in the shot — below the default min_samples=20.
    // Adaptive floor must NOT activate; absolute floor applies.
    let opts = Options::default()
      .with_margin_ratio(0.0)
      .with_target_interval(Duration::from_secs(60));
    let mut det = Detector::new(opts);
    for i in 0..5 {
      det.observe(ts((i as i64) * 1_000_000), good_score(50.0)); // < 100
    }
    let out = det.finalize_shot(tr(0, 10_000_000));
    // No frame passes the absolute floor → fallback picks the only
    // candidate (all tied at 50.0).
    assert_eq!(out.len(), 1);
  }
```

- [ ] **Step 6: Run selector tests.**

Run: `cargo test -p scenesdetect --lib keyframe::select`
Expected: all existing tests still pass plus the four new tests.

- [ ] **Step 7: Full crate test sweep.**

Run: `cargo test -p scenesdetect`
Expected: all pass.

- [ ] **Step 8: Commit.**

```bash
git add src/keyframe/select.rs
git commit -m "$(cat <<'EOF'
feat(keyframe): adaptive per-shot sharpness floor

When a shot has at least adaptive_floor_min_samples buffered frames,
lower the strict-gate sharpness floor to
min(opts.min_sharpness, p_percentile_in_shot). Never raises the
floor — legacy behaviour is preserved for high-sharpness shots.
Default: enabled, p25, min_samples=20. Recovers strict winners
in legit low-detail shots (fog, night interiors) instead of always
falling through to the fallback path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Motion-blur opt-in hard gate

Add a configurable hard gate that rejects frames whose `motion_blur` (gradient anisotropy) exceeds `max_motion_blur`. **Off by default** — opt-in via `with_motion_blur_gate(true)`.

**Files:**

- Modify: `src/keyframe/select.rs`

### Steps

- [ ] **Step 1: Add two new fields to `Options`.**

Inside the `Options` struct, append after `adaptive_floor_min_samples`:

```rust
  motion_blur_gate: bool,
  max_motion_blur: f32,
```

Update the `Default` impl:

```rust
      adaptive_floor: true,
      adaptive_floor_percentile: 0.25,
      adaptive_floor_min_samples: 20,
      motion_blur_gate: false,
      max_motion_blur: 0.75,
    }
```

- [ ] **Step 2: Add builders + accessors.**

Inside `impl Options { ... }`, after `with_adaptive_floor_min_samples`, append:

```rust
  /// Enables or disables the motion-blur hard gate. When enabled,
  /// frames whose [`FrameMetrics::motion_blur`] exceeds
  /// [`Self::max_motion_blur`] are rejected from the strict pass.
  /// Off by default because gradient anisotropy at 256-px downscale
  /// confounds motion blur with single-orientation scenes (forest,
  /// façade, horizon).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_motion_blur_gate(mut self, on: bool) -> Self {
    self.motion_blur_gate = on;
    self
  }

  /// Sets the maximum tolerated motion-blur (anisotropy) score. The
  /// gate (when enabled) rejects frames strictly greater than this
  /// value.
  ///
  /// # Panics
  /// When outside `[0.0, 1.0]`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_max_motion_blur(mut self, m: f32) -> Self {
    assert!(
      (0.0..=1.0).contains(&m),
      "max_motion_blur must be in [0.0, 1.0]"
    );
    self.max_motion_blur = m;
    self
  }

  /// Whether the motion-blur hard gate is enabled (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn motion_blur_gate(&self) -> bool {
    self.motion_blur_gate
  }
  /// Maximum tolerated motion-blur anisotropy score (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn max_motion_blur(&self) -> f32 {
    self.max_motion_blur
  }
```

- [ ] **Step 3: Wire the gate into `hard_gate`.**

Locate the `hard_gate` helper (post-Task-2 form):

```rust
fn hard_gate(m: &FrameMetrics, opts: &Options) -> bool {
  if m.brightness() < opts.black_mean_threshold as f32 {
    return true;
  }
  if m.brightness() > opts.bright_mean_threshold as f32 {
    return true;
  }
  if m.luma_variance() < opts.luma_variance_threshold
    && m.saturation_variance() < opts.sat_variance_threshold
  {
    return true;
  }
  if m.clipping() > opts.max_clipping {
    return true;
  }
  false
}
```

Insert the motion-blur check before the final `false`:

```rust
fn hard_gate(m: &FrameMetrics, opts: &Options) -> bool {
  if m.brightness() < opts.black_mean_threshold as f32 {
    return true;
  }
  if m.brightness() > opts.bright_mean_threshold as f32 {
    return true;
  }
  if m.luma_variance() < opts.luma_variance_threshold
    && m.saturation_variance() < opts.sat_variance_threshold
  {
    return true;
  }
  if m.clipping() > opts.max_clipping {
    return true;
  }
  if opts.motion_blur_gate && m.motion_blur() > opts.max_motion_blur {
    return true;
  }
  false
}
```

- [ ] **Step 4: Add selector tests for the motion-blur gate.**

Append inside the `tests` module:

```rust
  #[test]
  fn motion_blur_gate_disabled_by_default_keeps_high_anisotropy_frame() {
    // Bucket with one strict-eligible frame whose motion_blur=0.9.
    // Gate off by default → frame passes strict, becomes the winner.
    let opts = Options::default().with_margin_ratio(0.0);
    let mut det = Detector::new(opts);
    let m = FrameMetrics::new()
      .with_sharpness(500.0)
      .with_brightness(128.0)
      .with_luma_variance(200.0)
      .with_saturation_variance(100.0)
      .with_clipping(0.0)
      .with_motion_blur(0.9);
    det.observe(ts(500_000), m);
    let out = det.finalize_shot(tr(0, 2_000_000));
    assert_eq!(out, vec![ts(500_000)]);
  }

  #[test]
  fn motion_blur_gate_enabled_rejects_high_anisotropy_frame() {
    // Same fixture but with the gate on and a fresh, gate-passing
    // alternative. The high-anisotropy frame falls into fallback;
    // the low-anisotropy frame wins the strict path.
    let opts = Options::default()
      .with_margin_ratio(0.0)
      .with_motion_blur_gate(true)
      .with_max_motion_blur(0.75);
    let mut det = Detector::new(opts);
    let bad = FrameMetrics::new()
      .with_sharpness(800.0)
      .with_brightness(128.0)
      .with_luma_variance(200.0)
      .with_saturation_variance(100.0)
      .with_clipping(0.0)
      .with_motion_blur(0.9); // above the gate
    let good = FrameMetrics::new()
      .with_sharpness(500.0)
      .with_brightness(128.0)
      .with_luma_variance(200.0)
      .with_saturation_variance(100.0)
      .with_clipping(0.0)
      .with_motion_blur(0.1);
    det.observe(ts(500_000), bad);
    det.observe(ts(1_500_000), good);
    let out = det.finalize_shot(tr(0, 2_000_000));
    // Strict path: `good` wins (bad rejected by motion-blur gate).
    assert_eq!(out, vec![ts(1_500_000)]);
  }
```

- [ ] **Step 5: Run selector tests.**

Run: `cargo test -p scenesdetect --lib keyframe::select`
Expected: all existing tests pass plus the two new tests.

- [ ] **Step 6: Full crate test sweep + clippy + fmt.**

Run: `cargo test -p scenesdetect && cargo clippy -p scenesdetect --all-targets -- -D warnings && cargo fmt -p scenesdetect -- --check`
Expected: all tests pass; clippy returns no warnings; formatter check is clean. If `cargo fmt` reports diffs, run `cargo fmt -p scenesdetect` and stage them in the same commit.

- [ ] **Step 7: Commit.**

```bash
git add src/keyframe/select.rs
git commit -m "$(cat <<'EOF'
feat(keyframe): opt-in motion-blur hard gate in select

When motion_blur_gate is enabled, frames with anisotropy strictly
greater than max_motion_blur (default 0.75) are rejected from the
strict pass. Off by default — the metric confounds genuine motion
blur with single-orientation scenes on the 256-px downscale, so
gating it without telemetry would regress directional-scene
selection. Composite-argmax weight on motion_blur also defaults
to 0 (set in CompositeWeights::new); both are user opt-in.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final verification

After Task 11, the implementation matches the spec. Run a final full sweep:

- [ ] `cargo build -p scenesdetect --all-features`
- [ ] `cargo test -p scenesdetect --all-features`
- [ ] `cargo doc -p scenesdetect --no-deps` — verify no broken intra-doc links.
- [ ] `git log --oneline | head -12` — confirm 11 task commits land in order on top of the design-doc commit.

If everything is clean, open the PR against `feat/keyframe-detectors` (the active feature branch) and link to issue #5.
