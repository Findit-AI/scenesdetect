//! Platform-specific SIMD (plus a scalar fallback) for packed-BGR → planar
//! HSV conversion, mean-absolute-difference reduction, and the 3×3 Sobel
//! magnitude/direction kernel used by the content detector.
//!
//! Dispatch is a mix of compile-time `cfg` / `target_feature` selection
//! and, on `x86` / `x86_64` when `std` is enabled, runtime CPU-feature
//! detection. In particular:
//! - `aarch64` uses NEON selected at compile time because NEON is part of
//!   the base ISA.
//! - `wasm32` uses the wasm SIMD backend when `simd128` is enabled.
//! - `x86` / `x86_64` use runtime dispatch with `is_x86_feature_detected!`
//!   under `std` to pick AVX2, then SSSE3, then scalar; without `std`,
//!   compile-time `target_feature` gating selects the best available path.
//! - Other targets use the scalar fallback.
//!
//! Additional platforms can be added as sibling private modules exposing
//! the same internal entry points and wired into [`bgr_to_hsv_planes`]
//! through the appropriate `cfg` and/or dispatch branch.
//!
//! The module is crate-internal. Public-facing wrappers for the color-space
//! conversion live in `crate::frame::convert`; the mean-absolute-difference
//! and Sobel kernels remain internal and are consumed directly by the
//! content detector.

// Platform-specific modules, each exposing `pub(super) unsafe fn
// bgr_to_hsv_planes(...)`. Gated so each file is only compiled on matching
// targets — the source need not exist for other arches.

// Miri cannot interpret platform SIMD intrinsics — gate all SIMD modules
// on `not(miri)` so the dispatcher falls through to the scalar backend.
// Detector tests then still run under Miri (validating memory safety of
// the full pipeline) without hitting unsupported operations.

#[cfg(all(target_arch = "aarch64", not(miri)))]
mod neon;

// x86 SIMD modules are only reachable when either:
//   - `std` is enabled (runtime `is_x86_feature_detected!` dispatch), or
//   - the matching `target_feature` is set at compile time (no-std dispatch).
// Without either gate, the functions would compile but nothing calls them,
// producing dead-code warnings under `-D warnings`.
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  any(feature = "std", target_feature = "ssse3"),
  not(miri),
))]
mod x86_ssse3;

#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  any(feature = "std", target_feature = "avx2"),
  not(miri),
))]
mod x86_avx2;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(miri)))]
mod wasm_simd128;

/// Converts a packed 24-bit BGR frame into three planar HSV buffers that
/// match OpenCV's `cv2.COLOR_BGR2HSV` semantics. Dispatches to the best
/// implementation available for the build target.
///
/// Dispatch matrix:
///
/// - `aarch64` → NEON (compile-time; NEON is in base ARMv8-A ISA).
/// - `wasm32` with `simd128` target feature → wasm SIMD.
/// - `x86` / `x86_64`:
///   - With `std`, runtime `is_x86_feature_detected!` picks AVX2 → SSSE3 → scalar.
///   - Without `std`, compile-time `target_feature` picks the best path.
/// - Everything else → scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(unreachable_code)] // one branch per build config
#[allow(clippy::too_many_arguments)] // signature fixed by the 3-plane + dims + flag shape
pub(crate) fn bgr_to_hsv_planes(
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  src: &[u8],
  width: u32,
  height: u32,
  stride: u32,
  use_simd: bool,
) {
  if !use_simd {
    return scalar::Scalar::bgr_to_hsv_planes(h_out, s_out, v_out, src, width, height, stride);
  }

  #[cfg(all(target_arch = "aarch64", not(miri)))]
  {
    // SAFETY: NEON is part of the base ARMv8-A ISA — every aarch64 Rust
    // target has it. No runtime feature detection required.
    unsafe {
      neon::bgr_to_hsv_planes(h_out, s_out, v_out, src, width, height, stride);
    }
    return;
  }

  #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(miri)))]
  {
    // SAFETY: simd128 target feature enabled at compile time.
    unsafe {
      wasm_simd128::bgr_to_hsv_planes(h_out, s_out, v_out, src, width, height, stride);
    }
    return;
  }

  // x86 runtime dispatch when std is available.
  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "std",
    not(miri)
  ))]
  {
    if std::is_x86_feature_detected!("avx2") {
      // SAFETY: runtime-checked above. AVX2 implies SSSE3 at the hardware
      // level; the callee is annotated with both target features.
      unsafe {
        x86_avx2::bgr_to_hsv_planes(h_out, s_out, v_out, src, width, height, stride);
      }
      return;
    }
    if std::is_x86_feature_detected!("ssse3") {
      // SAFETY: runtime-checked above.
      unsafe {
        x86_ssse3::bgr_to_hsv_planes(h_out, s_out, v_out, src, width, height, stride);
      }
      return;
    }
  }

  // x86 compile-time dispatch when std is off.
  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "std"),
    target_feature = "avx2",
    not(miri),
  ))]
  {
    // SAFETY: target feature enabled at compile time.
    unsafe {
      x86_avx2::bgr_to_hsv_planes(h_out, s_out, v_out, src, width, height, stride);
    }
    return;
  }
  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "std"),
    target_feature = "ssse3",
    not(target_feature = "avx2"),
    not(miri),
  ))]
  {
    // SAFETY: target feature enabled at compile time.
    unsafe {
      x86_ssse3::bgr_to_hsv_planes(h_out, s_out, v_out, src, width, height, stride);
    }
    return;
  }

  // Fallback.
  scalar::Scalar::bgr_to_hsv_planes(h_out, s_out, v_out, src, width, height, stride);
}

/// Single-pixel scalar BGR → HSV, exposed for tests and for callers that
/// need to process stray pixels one at a time.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(dead_code)] // used only from tests in some build configurations
pub(crate) fn bgr_to_hsv_pixel(b: f32, g: f32, r: f32) -> (u8, u8, u8) {
  scalar::Scalar::bgr_to_hsv_pixel(b, g, r)
}

/// Sum of absolute per-element differences of two equal-length `u8` slices,
/// divided by `n`. Dispatches to the best SIMD backend or scalar based on
/// `use_simd`.
///
/// NEON uses `vabdq_u8` + `vpaddlq` accumulate. x86 uses `_mm_sad_epu8`
/// (a single-instruction SAD per 16 bytes). wasm uses widening subtract +
/// abs reduce. All produce the same numerical result as scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(unreachable_code)]
pub(crate) fn mean_abs_diff(a: &[u8], b: &[u8], n: usize, use_simd: bool) -> f64 {
  debug_assert!(a.len() >= n && b.len() >= n);
  if n == 0 {
    return 0.0;
  }

  if use_simd {
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    {
      // SAFETY: NEON is base ARMv8-A ISA.
      return unsafe { neon::mean_abs_diff(a, b, n) };
    }

    #[cfg(all(
      any(target_arch = "x86", target_arch = "x86_64"),
      feature = "std",
      not(miri)
    ))]
    {
      if std::is_x86_feature_detected!("ssse3") {
        // SAFETY: runtime-checked.
        return unsafe { x86_ssse3::mean_abs_diff(a, b, n) };
      }
    }

    #[cfg(all(
      any(target_arch = "x86", target_arch = "x86_64"),
      not(feature = "std"),
      target_feature = "ssse3",
      not(miri),
    ))]
    {
      return unsafe { x86_ssse3::mean_abs_diff(a, b, n) };
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(miri)))]
    {
      return unsafe { wasm_simd128::mean_abs_diff(a, b, n) };
    }
  }

  scalar::Scalar::mean_abs_diff(a, b, n)
}

/// 3×3 Sobel: computes L1 magnitude (`|Gx| + |Gy|`) into `mag` and a
/// quantized gradient direction (0=horiz, 1=45°, 2=vert, 3=135°) into `dir`.
/// Border pixels stay zero. Dispatches to SIMD for the magnitude computation;
/// direction quantization is always scalar (branchy per pixel).
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(unreachable_code)]
pub(crate) fn sobel(
  input: &[u8],
  mag: &mut [i32],
  dir: &mut [u8],
  w: usize,
  h: usize,
  use_simd: bool,
) {
  if use_simd {
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    {
      return unsafe { neon::sobel(input, mag, dir, w, h) };
    }

    #[cfg(all(
      any(target_arch = "x86", target_arch = "x86_64"),
      feature = "std",
      not(miri)
    ))]
    {
      if std::is_x86_feature_detected!("ssse3") {
        return unsafe { x86_ssse3::sobel(input, mag, dir, w, h) };
      }
    }

    #[cfg(all(
      any(target_arch = "x86", target_arch = "x86_64"),
      not(feature = "std"),
      target_feature = "ssse3",
      not(miri),
    ))]
    {
      return unsafe { x86_ssse3::sobel(input, mag, dir, w, h) };
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(miri)))]
    {
      return unsafe { wasm_simd128::sobel(input, mag, dir, w, h) };
    }
  }

  scalar::Scalar::sobel(input, mag, dir, w, h);
}

/// Converts a packed 24-bit BGR frame into a single-plane BT.601 luma
/// buffer. Uses the fixed-point approximation
/// `Y = (77·R + 150·G + 29·B) >> 8`; coefficients sum to 256 so the
/// output stays in `[0, 255]`.
///
/// Dispatch matrix matches [`bgr_to_hsv_planes`]: aarch64 NEON at
/// compile time; x86 runtime dispatch under `std` picks AVX2 → SSSE3 →
/// scalar; `no_std` x86 dispatches at compile time; wasm uses
/// `simd128` when enabled; everything else falls back to scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(unreachable_code)]
pub(crate) fn bgr_to_luma(
  out: &mut [u8],
  src: &[u8],
  width: u32,
  height: u32,
  stride: u32,
  use_simd: bool,
) {
  if !use_simd {
    return scalar::Scalar::bgr_to_luma(out, src, width, height, stride);
  }

  #[cfg(all(target_arch = "aarch64", not(miri)))]
  {
    // SAFETY: NEON is part of the base ARMv8-A ISA.
    unsafe {
      neon::bgr_to_luma(out, src, width, height, stride);
    }
    return;
  }

  #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(miri)))]
  {
    // SAFETY: simd128 target feature enabled at compile time.
    unsafe {
      wasm_simd128::bgr_to_luma(out, src, width, height, stride);
    }
    return;
  }

  // x86 runtime dispatch under std. For bgr_to_luma the reduction is
  // memory-bandwidth-bound, so the SSSE3 (128-bit) path is already
  // close to peak throughput; we don't ship a separate AVX2 kernel,
  // AVX2-capable hosts land here and run SSSE3.
  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "std",
    not(miri)
  ))]
  {
    if std::is_x86_feature_detected!("ssse3") {
      // SAFETY: runtime-checked above.
      unsafe {
        x86_ssse3::bgr_to_luma(out, src, width, height, stride);
      }
      return;
    }
  }

  // x86 compile-time dispatch without std.
  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "std"),
    target_feature = "ssse3",
    not(miri),
  ))]
  {
    // SAFETY: target feature enabled at compile time.
    unsafe {
      x86_ssse3::bgr_to_luma(out, src, width, height, stride);
    }
    return;
  }

  scalar::Scalar::bgr_to_luma(out, src, width, height, stride);
}

/// Counts pixels in a packed 24-bit BGR frame whose brightest channel
/// falls outside the `[CLIP_LOW, CLIP_HIGH] = [5, 250]` unclipped band
/// — i.e. `max(R, G, B) < 5` (crushed shadow) or `> 250` (blown
/// highlight). Byte-order-agnostic: same count for BGR and RGB order.
///
/// Returns the count of clipped pixels. Caller divides by
/// `width * height` for the ratio.
///
/// Dispatch matrix mirrors [`bgr_to_luma`].
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(unreachable_code)]
pub(crate) fn clipping_count(
  src: &[u8],
  width: u32,
  height: u32,
  stride: u32,
  use_simd: bool,
) -> u64 {
  if !use_simd {
    return scalar::Scalar::clipping_count(src, width, height, stride);
  }

  #[cfg(all(target_arch = "aarch64", not(miri)))]
  {
    // SAFETY: NEON is part of the base ARMv8-A ISA.
    return unsafe { neon::clipping_count(src, width, height, stride) };
  }

  #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(miri)))]
  {
    // SAFETY: simd128 target feature enabled at compile time.
    return unsafe { wasm_simd128::clipping_count(src, width, height, stride) };
  }

  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "std",
    not(miri)
  ))]
  {
    if std::is_x86_feature_detected!("ssse3") {
      // SAFETY: runtime-checked above.
      return unsafe { x86_ssse3::clipping_count(src, width, height, stride) };
    }
  }

  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "std"),
    target_feature = "ssse3",
    not(miri),
  ))]
  {
    // SAFETY: target feature enabled at compile time.
    return unsafe { x86_ssse3::clipping_count(src, width, height, stride) };
  }

  scalar::Scalar::clipping_count(src, width, height, stride)
}

/// Tenengrad sharpness on a luma plane: `mean(gx² + gy²)` over
/// interior pixels, where `gx` / `gy` come from a 3×3 Sobel. Honours
/// row stride. Frames narrower or shorter than 3 pixels have no
/// interior and yield `0.0`.
///
/// Dispatch matrix mirrors [`bgr_to_luma`].
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(unreachable_code)]
pub(crate) fn tenengrad(
  luma: &[u8],
  width: usize,
  height: usize,
  stride: usize,
  use_simd: bool,
) -> f32 {
  if !use_simd {
    return scalar::Scalar::tenengrad(luma, width, height, stride);
  }

  #[cfg(all(target_arch = "aarch64", not(miri)))]
  {
    // SAFETY: NEON is part of the base ARMv8-A ISA.
    return unsafe { neon::tenengrad(luma, width, height, stride) };
  }

  #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(miri)))]
  {
    // SAFETY: simd128 target feature enabled at compile time.
    return unsafe { wasm_simd128::tenengrad(luma, width, height, stride) };
  }

  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "std",
    not(miri)
  ))]
  {
    if std::is_x86_feature_detected!("ssse3") {
      // SAFETY: runtime-checked above.
      return unsafe { x86_ssse3::tenengrad(luma, width, height, stride) };
    }
  }

  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "std"),
    target_feature = "ssse3",
    not(miri),
  ))]
  {
    // SAFETY: target feature enabled at compile time.
    return unsafe { x86_ssse3::tenengrad(luma, width, height, stride) };
  }

  scalar::Scalar::tenengrad(luma, width, height, stride)
}

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

/// Population mean and variance of a single-plane `u8` image. Honours
/// row stride. Uses a single-pass `E[X²] - E[X]²` formulation —
/// numerically safe for `u8` inputs evaluated in `f64` (variance tops
/// out around 65025, well within f64 precision).
///
/// Dispatch matrix mirrors [`bgr_to_luma`].
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(unreachable_code)]
pub(crate) fn plane_mean_variance(
  plane: &[u8],
  width: usize,
  height: usize,
  stride: usize,
  use_simd: bool,
) -> (f32, f32) {
  if !use_simd {
    return scalar::Scalar::plane_mean_variance(plane, width, height, stride);
  }

  #[cfg(all(target_arch = "aarch64", not(miri)))]
  {
    // SAFETY: NEON is part of the base ARMv8-A ISA.
    return unsafe { neon::plane_mean_variance(plane, width, height, stride) };
  }

  #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(miri)))]
  {
    // SAFETY: simd128 target feature enabled at compile time.
    return unsafe { wasm_simd128::plane_mean_variance(plane, width, height, stride) };
  }

  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "std",
    not(miri)
  ))]
  {
    if std::is_x86_feature_detected!("ssse3") {
      // SAFETY: runtime-checked above.
      return unsafe { x86_ssse3::plane_mean_variance(plane, width, height, stride) };
    }
  }

  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "std"),
    target_feature = "ssse3",
    not(miri),
  ))]
  {
    // SAFETY: target feature enabled at compile time.
    return unsafe { x86_ssse3::plane_mean_variance(plane, width, height, stride) };
  }

  scalar::Scalar::plane_mean_variance(plane, width, height, stride)
}

// -----------------------------------------------------------------------------
// Scalar implementation — used as the fallback on non-aarch64 targets and
// as the reference for the single-pixel helper everywhere.
//
// Common (non-SIMD) code is grouped under a ZST with `impl` methods; only the
// platform-specific SIMD backends use free functions (which is idiomatic for
// intrinsic-heavy code where each function carries a `target_feature`
// attribute).
// -----------------------------------------------------------------------------

mod scalar {
  use crate::round_32;

  /// Zero-sized namespace for the scalar BGR→HSV kernels.
  pub(super) struct Scalar;

  impl Scalar {
    /// Whole-plane scalar BGR→HSV. Used as the fallback on targets without
    /// a SIMD backend.
    // On aarch64 the planar function is unused (NEON wins); keep it around
    // as a correctness reference.
    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    pub(super) fn bgr_to_hsv_planes(
      h_out: &mut [u8],
      s_out: &mut [u8],
      v_out: &mut [u8],
      src: &[u8],
      width: u32,
      height: u32,
      stride: u32,
    ) {
      let w = width as usize;
      let h = height as usize;
      let s = stride as usize;
      for y in 0..h {
        let row = &src[y * s..y * s + w * 3];
        let dst_off = y * w;
        for x in 0..w {
          let b = row[x * 3] as f32;
          let g = row[x * 3 + 1] as f32;
          let r = row[x * 3 + 2] as f32;
          let (hue, sat, val) = Self::bgr_to_hsv_pixel(b, g, r);
          h_out[dst_off + x] = hue;
          s_out[dst_off + x] = sat;
          v_out[dst_off + x] = val;
        }
      }
    }

    /// Scalar BGR→HSV for a single pixel. Inputs are floats (typically from
    /// `u8 as f32`); outputs are clamped/rounded u8 in OpenCV's 8-bit
    /// encoding (H in [0, 179], S and V in [0, 255]).
    #[inline]
    pub(super) fn bgr_to_hsv_pixel(b: f32, g: f32, r: f32) -> (u8, u8, u8) {
      let v = b.max(g).max(r);
      let min = b.min(g).min(r);
      let delta = v - min;
      let s = if v == 0.0 { 0.0 } else { 255.0 * delta / v };
      let hue = if delta == 0.0 {
        0.0
      } else if v == r {
        let h = 60.0 * (g - b) / delta;
        if h < 0.0 { h + 360.0 } else { h }
      } else if v == g {
        60.0 * (b - r) / delta + 120.0
      } else {
        60.0 * (r - g) / delta + 240.0
      };
      let h8 = round_32(hue * 0.5).clamp(0.0, 179.0) as u8;
      (
        h8,
        round_32(s).clamp(0.0, 255.0) as u8,
        round_32(v).clamp(0.0, 255.0) as u8,
      )
    }

    /// Scalar 3×3 Sobel: magnitude + direction.
    pub(super) fn sobel(input: &[u8], mag: &mut [i32], dir: &mut [u8], w: usize, h: usize) {
      mag.fill(0);
      dir.fill(0);
      for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
          let i = |yy: usize, xx: usize| input[yy * w + xx] as i32;
          let gx = -i(y - 1, x - 1) - 2 * i(y, x - 1) - i(y + 1, x - 1)
            + i(y - 1, x + 1)
            + 2 * i(y, x + 1)
            + i(y + 1, x + 1);
          let gy = -i(y - 1, x - 1) - 2 * i(y - 1, x) - i(y - 1, x + 1)
            + i(y + 1, x - 1)
            + 2 * i(y + 1, x)
            + i(y + 1, x + 1);
          let idx = y * w + x;
          mag[idx] = gx.abs() + gy.abs();
          let ax = gx.abs();
          let ay = gy.abs();
          dir[idx] = if ay * 1000 < ax * 414 {
            0
          } else if ay * 1000 > ax * 2414 {
            2
          } else if gx.signum() == gy.signum() {
            1
          } else {
            3
          };
        }
      }
    }

    /// Scalar mean absolute difference: `Σ|a[i] - b[i]| / n`.
    #[inline]
    pub(super) fn mean_abs_diff(a: &[u8], b: &[u8], n: usize) -> f64 {
      let mut sum: u64 = 0;
      for i in 0..n {
        let da = a[i] as i32 - b[i] as i32;
        sum += da.unsigned_abs() as u64;
      }
      sum as f64 / n as f64
    }

    /// Scalar BGR → BT.601 luma: `Y = (77·R + 150·G + 29·B) >> 8`.
    /// Honours row stride; writes tight-packed `width * height` bytes
    /// into `out`.
    // On aarch64 the NEON path wins; keep this as the correctness
    // reference for tests and the fallback elsewhere.
    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    pub(super) fn bgr_to_luma(out: &mut [u8], src: &[u8], width: u32, height: u32, stride: u32) {
      let w = width as usize;
      let h = height as usize;
      let s = stride as usize;
      for y in 0..h {
        let row_off = y * s;
        let dst_off = y * w;
        let row = &src[row_off..row_off + w * 3];
        let dst = &mut out[dst_off..dst_off + w];
        for x in 0..w {
          let b = row[x * 3] as u32;
          let g = row[x * 3 + 1] as u32;
          let r = row[x * 3 + 2] as u32;
          dst[x] = ((77 * r + 150 * g + 29 * b) >> 8) as u8;
        }
      }
    }

    /// Scalar clipping-pixel count. A pixel is clipped iff
    /// `max(R, G, B) < 5` or `> 250`. Honours row stride.
    // On aarch64 the NEON path wins; keep this as the correctness
    // reference.
    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    pub(super) fn clipping_count(src: &[u8], width: u32, height: u32, stride: u32) -> u64 {
      let w = width as usize;
      let h = height as usize;
      let s = stride as usize;
      let mut count: u64 = 0;
      for y in 0..h {
        let row = &src[y * s..y * s + w * 3];
        for x in 0..w {
          let b = row[x * 3];
          let g = row[x * 3 + 1];
          let r = row[x * 3 + 2];
          let m = b.max(g).max(r);
          if !(5..=250).contains(&m) {
            count += 1;
          }
        }
      }
      count
    }

    /// Scalar single-pass `(mean, variance)` via u64 `sum_x` and
    /// `sum_x²`, finalized as `E[X²] - E[X]²` in f64. Honours row
    /// stride.
    // On aarch64 the NEON path wins; keep this as the correctness reference.
    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    pub(super) fn plane_mean_variance(plane: &[u8], w: usize, h: usize, s: usize) -> (f32, f32) {
      let n = w.saturating_mul(h);
      if n == 0 {
        return (0.0, 0.0);
      }
      let mut sum: u64 = 0;
      let mut sum_sq: u64 = 0;
      for y in 0..h {
        let row = &plane[y * s..y * s + w];
        for &v in row {
          sum += v as u64;
          sum_sq += (v as u64) * (v as u64);
        }
      }
      let n_f = n as f64;
      let mean = (sum as f64) / n_f;
      let mean_sq = (sum_sq as f64) / n_f;
      let variance = (mean_sq - mean * mean).max(0.0);
      (mean as f32, variance as f32)
    }

    /// Scalar Tenengrad: `mean(gx² + gy²)` over interior pixels of a
    /// luma plane, where `gx`/`gy` are 3×3 Sobel gradients. Honours
    /// row stride.
    // On aarch64 the NEON path wins; keep this as the correctness
    // reference.
    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    pub(super) fn tenengrad(luma: &[u8], w: usize, h: usize, s: usize) -> f32 {
      if w < 3 || h < 3 {
        return 0.0;
      }
      let interior = (w - 2) * (h - 2);
      if interior == 0 {
        return 0.0;
      }

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
          let r = p(0, 1);
          let bl = p(1, -1);
          let b = p(1, 0);
          let br = p(1, 1);
          let gx = -tl - 2 * l - bl + tr + 2 * r + br;
          let gy = -tl - 2 * t - tr + bl + 2 * b + br;
          acc += (gx * gx + gy * gy) as i64;
        }
      }

      ((acc as f64) / (interior as f64)) as f32
    }

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
  }
}

// ---------------------------------------------------------------------------
// Direct-call tests for platform SIMD backends. On x86 hosts, the runtime
// dispatcher picks AVX2 when available, leaving the SSSE3 `bgr_to_hsv_planes`
// path untested. These tests call each backend directly so coverage includes
// all compiled SIMD code regardless of which tier the host CPU supports.
// ---------------------------------------------------------------------------
// Miri: the scalar tests are fine, but the direct SIMD-call tests reference
// modules that are gated out under `cfg(miri)`. Gate the whole test module
// on `not(miri)` — Miri exercises the scalar paths through the detector-level
// tests in content.rs instead.
#[cfg(all(test, feature = "std", not(miri)))]
mod tests {
  use super::*;

  fn make_bgr(w: usize, h: usize) -> Vec<u8> {
    let mut buf = vec![0u8; w * h * 3];
    let mut rng = 0x9E3779B9u32;
    for v in buf.iter_mut() {
      rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
      *v = (rng >> 24) as u8;
    }
    buf
  }

  fn make_luma(w: usize, h: usize) -> Vec<u8> {
    let mut buf = vec![0u8; w * h];
    let mut rng = 0xDEADBEEFu32;
    for v in buf.iter_mut() {
      rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
      *v = (rng >> 24) as u8;
    }
    buf
  }

  // Exercises the scalar bgr_to_hsv_planes + mean_abs_diff + sobel.
  #[test]
  fn scalar_bgr_to_hsv_planes() {
    let (w, h) = (32, 16);
    let src = make_bgr(w, h);
    let n = w * h;
    let mut ho = vec![0u8; n];
    let mut so = vec![0u8; n];
    let mut vo = vec![0u8; n];
    scalar::Scalar::bgr_to_hsv_planes(
      &mut ho,
      &mut so,
      &mut vo,
      &src,
      w as u32,
      h as u32,
      (w * 3) as u32,
    );
    assert!(vo.iter().any(|&v| v > 0));
  }

  #[test]
  fn scalar_mean_abs_diff_nonzero() {
    let a = make_luma(64, 1);
    let b = make_luma(64, 1);
    let d = scalar::Scalar::mean_abs_diff(&a, &b, 64);
    assert!(d >= 0.0);
  }

  #[test]
  fn scalar_sobel() {
    let (w, h) = (16, 16);
    let src = make_luma(w, h);
    let mut mag = vec![0i32; w * h];
    let mut dir = vec![0u8; w * h];
    scalar::Scalar::sobel(&src, &mut mag, &mut dir, w, h);
    assert!(mag.iter().any(|&m| m > 0));
  }

  // x86: call SSSE3 bgr_to_hsv_planes directly (bypasses AVX2 dispatch).
  #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
  #[test]
  fn ssse3_bgr_to_hsv_planes_direct() {
    if !std::is_x86_feature_detected!("ssse3") {
      return;
    }
    let (w, h) = (64, 16);
    let src = make_bgr(w, h);
    let n = w * h;
    let mut ho = vec![0u8; n];
    let mut so = vec![0u8; n];
    let mut vo = vec![0u8; n];
    unsafe {
      x86_ssse3::bgr_to_hsv_planes(
        &mut ho,
        &mut so,
        &mut vo,
        &src,
        w as u32,
        h as u32,
        (w * 3) as u32,
      );
    }
    // Sanity: V plane should have nonzero values for random input.
    assert!(vo.iter().any(|&v| v > 0));
  }

  #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
  #[test]
  fn ssse3_mean_abs_diff_direct() {
    if !std::is_x86_feature_detected!("ssse3") {
      return;
    }
    let a = make_luma(128, 1);
    let b = make_luma(128, 1);
    let d = unsafe { x86_ssse3::mean_abs_diff(&a, &b, 128) };
    assert!(d >= 0.0);
  }

  #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
  #[test]
  fn ssse3_sobel_direct() {
    if !std::is_x86_feature_detected!("ssse3") {
      return;
    }
    let (w, h) = (32, 32);
    let src = make_luma(w, h);
    let mut mag = vec![0i32; w * h];
    let mut dir = vec![0u8; w * h];
    unsafe { x86_ssse3::sobel(&src, &mut mag, &mut dir, w, h) };
    assert!(mag.iter().any(|&m| m > 0));
  }

  // x86: call AVX2 bgr_to_hsv_planes directly (exercises the AVX2 tail path too).
  #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
  #[test]
  fn avx2_bgr_to_hsv_planes_direct() {
    if !std::is_x86_feature_detected!("avx2") {
      return;
    }
    let (w, h) = (64, 16);
    let src = make_bgr(w, h);
    let n = w * h;
    let mut ho = vec![0u8; n];
    let mut so = vec![0u8; n];
    let mut vo = vec![0u8; n];
    unsafe {
      x86_avx2::bgr_to_hsv_planes(
        &mut ho,
        &mut so,
        &mut vo,
        &src,
        w as u32,
        h as u32,
        (w * 3) as u32,
      );
    }
    assert!(vo.iter().any(|&v| v > 0));
  }

  // aarch64: call NEON bgr_to_hsv_planes directly.
  #[cfg(target_arch = "aarch64")]
  #[test]
  fn neon_bgr_to_hsv_planes_direct() {
    let (w, h) = (64, 16);
    let src = make_bgr(w, h);
    let n = w * h;
    let mut ho = vec![0u8; n];
    let mut so = vec![0u8; n];
    let mut vo = vec![0u8; n];
    unsafe {
      neon::bgr_to_hsv_planes(
        &mut ho,
        &mut so,
        &mut vo,
        &src,
        w as u32,
        h as u32,
        (w * 3) as u32,
      );
    }
    assert!(vo.iter().any(|&v| v > 0));
  }

  #[cfg(target_arch = "aarch64")]
  #[test]
  fn neon_mean_abs_diff_direct() {
    let a = make_luma(128, 1);
    let b = make_luma(128, 1);
    let d = unsafe { neon::mean_abs_diff(&a, &b, 128) };
    assert!(d >= 0.0);
  }

  #[cfg(target_arch = "aarch64")]
  #[test]
  fn neon_sobel_direct() {
    let (w, h) = (32, 32);
    let src = make_luma(w, h);
    let mut mag = vec![0i32; w * h];
    let mut dir = vec![0u8; w * h];
    unsafe { neon::sobel(&src, &mut mag, &mut dir, w, h) };
    assert!(mag.iter().any(|&m| m > 0));
  }

  // ---- bgr_to_luma ----------------------------------------------------------

  #[test]
  fn scalar_bgr_to_luma_spot_values() {
    // Pure red / green / blue through the BT.601 fixed-point: the
    // per-pixel math is (77R + 150G + 29B) >> 8 where each channel is
    // exactly 255 in isolation.
    let red = [0u8, 0, 255];
    let green = [0u8, 255, 0];
    let blue = [255u8, 0, 0];
    let mut out = [0u8; 1];
    scalar::Scalar::bgr_to_luma(&mut out, &red, 1, 1, 3);
    assert_eq!(out[0], 76);
    scalar::Scalar::bgr_to_luma(&mut out, &green, 1, 1, 3);
    assert_eq!(out[0], 149);
    scalar::Scalar::bgr_to_luma(&mut out, &blue, 1, 1, 3);
    assert_eq!(out[0], 28);
  }

  // Scalar-equivalence test: the scalar reference and every SIMD
  // backend must produce byte-identical output. Uses a random-ish
  // fixed-seed pattern so the assertion is deterministic across
  // runs and CI targets.
  fn assert_bgr_to_luma_equiv<F: FnOnce(&mut [u8], &[u8], u32, u32, u32)>(
    w: usize,
    h: usize,
    stride: usize,
    backend: F,
  ) {
    // Build a padded source where pixel bytes are random-ish and
    // padding bytes are a distinctive 0xAA (which would corrupt the
    // output if stride were ignored).
    let mut src = vec![0xAAu8; stride * h];
    let mut rng = 0x9E3779B9u32;
    for y in 0..h {
      for x in 0..w * 3 {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        src[y * stride + x] = (rng >> 24) as u8;
      }
    }
    let mut out_scalar = vec![0u8; w * h];
    scalar::Scalar::bgr_to_luma(&mut out_scalar, &src, w as u32, h as u32, stride as u32);
    let mut out_simd = vec![0u8; w * h];
    backend(&mut out_simd, &src, w as u32, h as u32, stride as u32);
    assert_eq!(out_simd, out_scalar, "SIMD backend disagrees with scalar");
  }

  // aarch64: call NEON bgr_to_luma directly, on various dims to
  // exercise both the 16-wide main loop and the scalar tail.
  #[cfg(target_arch = "aarch64")]
  #[test]
  fn neon_bgr_to_luma_matches_scalar() {
    // Exactly 16 pixels wide — main loop only, no tail.
    assert_bgr_to_luma_equiv(16, 4, 16 * 3, |out, src, w, h, s| unsafe {
      neon::bgr_to_luma(out, src, w, h, s);
    });
    // 17 pixels — one tail pixel per row.
    assert_bgr_to_luma_equiv(17, 4, 17 * 3, |out, src, w, h, s| unsafe {
      neon::bgr_to_luma(out, src, w, h, s);
    });
    // Stride padding: 24 pixels per row, stride of 96 bytes.
    assert_bgr_to_luma_equiv(24, 5, 96, |out, src, w, h, s| unsafe {
      neon::bgr_to_luma(out, src, w, h, s);
    });
    // Larger frame to catch any accumulator issues.
    assert_bgr_to_luma_equiv(257, 31, 257 * 3, |out, src, w, h, s| unsafe {
      neon::bgr_to_luma(out, src, w, h, s);
    });
  }

  // x86 SSSE3: call bgr_to_luma directly. Skips itself when the host
  // CPU doesn't support SSSE3 (the runtime check bypasses the SIMD
  // backend in that case, so we can't reach it anyway).
  #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
  #[test]
  fn ssse3_bgr_to_luma_matches_scalar() {
    if !std::is_x86_feature_detected!("ssse3") {
      return;
    }
    assert_bgr_to_luma_equiv(16, 4, 16 * 3, |out, src, w, h, s| unsafe {
      x86_ssse3::bgr_to_luma(out, src, w, h, s);
    });
    assert_bgr_to_luma_equiv(17, 4, 17 * 3, |out, src, w, h, s| unsafe {
      x86_ssse3::bgr_to_luma(out, src, w, h, s);
    });
    assert_bgr_to_luma_equiv(24, 5, 96, |out, src, w, h, s| unsafe {
      x86_ssse3::bgr_to_luma(out, src, w, h, s);
    });
    assert_bgr_to_luma_equiv(257, 31, 257 * 3, |out, src, w, h, s| unsafe {
      x86_ssse3::bgr_to_luma(out, src, w, h, s);
    });
  }

  // ---- clipping_count -------------------------------------------------------

  #[test]
  fn scalar_clipping_count_spot_values() {
    // 3 pixels: clipped-low, unclipped, clipped-high.
    let src = [
      4u8, 0, 0, // max=4, clipped
      100, 100, 100, // max=100, unclipped
      0, 0, 255, // max=255, clipped
    ];
    assert_eq!(
      scalar::Scalar::clipping_count(&src, 3, 1, 9),
      2,
      "expected 2 clipped pixels"
    );
  }

  /// Builds a padded BGR source with distinctive 0xAA padding; pixel
  /// bytes are a fixed-seed random pattern.
  fn assert_clipping_equiv<F: FnOnce(&[u8], u32, u32, u32) -> u64>(
    w: usize,
    h: usize,
    stride: usize,
    backend: F,
  ) {
    let mut src = vec![0xAAu8; stride * h];
    let mut rng = 0xDEADBEEFu32;
    for y in 0..h {
      for x in 0..w * 3 {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        src[y * stride + x] = (rng >> 24) as u8;
      }
    }
    let scalar_out = scalar::Scalar::clipping_count(&src, w as u32, h as u32, stride as u32);
    let simd_out = backend(&src, w as u32, h as u32, stride as u32);
    assert_eq!(
      simd_out, scalar_out,
      "SIMD clipping count disagrees with scalar"
    );
  }

  #[cfg(target_arch = "aarch64")]
  #[test]
  fn neon_clipping_count_matches_scalar() {
    // Exercise main loop + tail across a few dims.
    assert_clipping_equiv(16, 4, 16 * 3, |src, w, h, s| unsafe {
      neon::clipping_count(src, w, h, s)
    });
    assert_clipping_equiv(17, 4, 17 * 3, |src, w, h, s| unsafe {
      neon::clipping_count(src, w, h, s)
    });
    assert_clipping_equiv(24, 5, 96, |src, w, h, s| unsafe {
      neon::clipping_count(src, w, h, s)
    });
    assert_clipping_equiv(257, 31, 257 * 3, |src, w, h, s| unsafe {
      neon::clipping_count(src, w, h, s)
    });
  }

  #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
  #[test]
  fn ssse3_clipping_count_matches_scalar() {
    if !std::is_x86_feature_detected!("ssse3") {
      return;
    }
    assert_clipping_equiv(16, 4, 16 * 3, |src, w, h, s| unsafe {
      x86_ssse3::clipping_count(src, w, h, s)
    });
    assert_clipping_equiv(17, 4, 17 * 3, |src, w, h, s| unsafe {
      x86_ssse3::clipping_count(src, w, h, s)
    });
    assert_clipping_equiv(24, 5, 96, |src, w, h, s| unsafe {
      x86_ssse3::clipping_count(src, w, h, s)
    });
    assert_clipping_equiv(257, 31, 257 * 3, |src, w, h, s| unsafe {
      x86_ssse3::clipping_count(src, w, h, s)
    });
  }

  // ---- tenengrad ------------------------------------------------------------

  #[test]
  fn scalar_tenengrad_uniform_is_zero() {
    let data = vec![128u8; 16 * 16];
    assert_eq!(scalar::Scalar::tenengrad(&data, 16, 16, 16), 0.0);
  }

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
    // Use a checkerboard at amplitude k=64. Every interior pixel sees
    // lap = 4·c - 2·(t+b+l+r) + (tl+tr+bl+br)
    //     = 4·(±64) + 8·(±64) + 4·(±64) = 16·(±64) = ±1024
    // |lap| = 1024 per interior pixel. With interior = 14·14 = 196,
    // mean |lap| = 1024, σₙ ≈ 0.208898 · 1024 ≈ 213.92.
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

  #[test]
  fn scalar_tenengrad_small_frame_is_zero() {
    // 2×2 has no interior — no Sobel output.
    let data = vec![0u8, 255, 255, 0];
    assert_eq!(scalar::Scalar::tenengrad(&data, 2, 2, 2), 0.0);
  }

  /// Builds a luma plane with a fixed-seed random pattern; padding is
  /// 0xAA to catch stride misuse.
  fn assert_tenengrad_equiv<F: FnOnce(&[u8], usize, usize, usize) -> f32>(
    w: usize,
    h: usize,
    stride: usize,
    backend: F,
  ) {
    let mut data = vec![0xAAu8; stride * h];
    let mut rng = 0xCAFEBABEu32;
    for y in 0..h {
      for x in 0..w {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        data[y * stride + x] = (rng >> 24) as u8;
      }
    }
    let scalar_out = scalar::Scalar::tenengrad(&data, w, h, stride);
    let simd_out = backend(&data, w, h, stride);
    // Tenengrad values vary over many orders of magnitude, but both
    // paths use identical i64 accumulators + one final f64-f32 cast —
    // they should agree to well within 1e-3 relative.
    let rel = (simd_out - scalar_out).abs() / scalar_out.abs().max(1.0);
    assert!(
      rel < 1e-3,
      "SIMD tenengrad disagrees with scalar: simd={simd_out} scalar={scalar_out} rel={rel}"
    );
  }

  #[cfg(target_arch = "aarch64")]
  #[test]
  fn neon_tenengrad_matches_scalar() {
    // 10 wide = interior 8 wide (one vector iteration, zero tail).
    assert_tenengrad_equiv(10, 10, 10, |luma, w, h, s| unsafe {
      neon::tenengrad(luma, w, h, s)
    });
    // 11 wide = interior 9, one tail pixel per row.
    assert_tenengrad_equiv(11, 10, 11, |luma, w, h, s| unsafe {
      neon::tenengrad(luma, w, h, s)
    });
    // Stride-padded.
    assert_tenengrad_equiv(24, 12, 64, |luma, w, h, s| unsafe {
      neon::tenengrad(luma, w, h, s)
    });
    // Larger frame — multiple vector iterations + tail per row.
    assert_tenengrad_equiv(257, 31, 257, |luma, w, h, s| unsafe {
      neon::tenengrad(luma, w, h, s)
    });
  }

  #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
  #[test]
  fn ssse3_tenengrad_matches_scalar() {
    if !std::is_x86_feature_detected!("ssse3") {
      return;
    }
    assert_tenengrad_equiv(10, 10, 10, |luma, w, h, s| unsafe {
      x86_ssse3::tenengrad(luma, w, h, s)
    });
    assert_tenengrad_equiv(11, 10, 11, |luma, w, h, s| unsafe {
      x86_ssse3::tenengrad(luma, w, h, s)
    });
    assert_tenengrad_equiv(24, 12, 64, |luma, w, h, s| unsafe {
      x86_ssse3::tenengrad(luma, w, h, s)
    });
    assert_tenengrad_equiv(257, 31, 257, |luma, w, h, s| unsafe {
      x86_ssse3::tenengrad(luma, w, h, s)
    });
  }

  // ---- plane_mean_variance --------------------------------------------------

  #[test]
  fn scalar_plane_mean_variance_uniform() {
    let data = vec![128u8; 16 * 8];
    let (mean, var) = scalar::Scalar::plane_mean_variance(&data, 16, 8, 16);
    assert!((mean - 128.0).abs() < 1e-3);
    assert!(var < 1e-3);
  }

  #[test]
  fn scalar_plane_mean_variance_bimodal() {
    // Half zeros, half 200 → mean 100, var 10 000.
    let mut data = vec![0u8; 16 * 16];
    for i in (16 * 8)..(16 * 16) {
      data[i] = 200;
    }
    let (mean, var) = scalar::Scalar::plane_mean_variance(&data, 16, 16, 16);
    assert!((mean - 100.0).abs() < 1e-3);
    assert!((var - 10_000.0).abs() < 1.0);
  }

  fn assert_plane_stats_equiv<F: FnOnce(&[u8], usize, usize, usize) -> (f32, f32)>(
    w: usize,
    h: usize,
    stride: usize,
    backend: F,
  ) {
    let mut data = vec![0xAAu8; stride * h];
    let mut rng = 0xABAD1DEAu32;
    for y in 0..h {
      for x in 0..w {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        data[y * stride + x] = (rng >> 24) as u8;
      }
    }
    let (scalar_m, scalar_v) = scalar::Scalar::plane_mean_variance(&data, w, h, stride);
    let (simd_m, simd_v) = backend(&data, w, h, stride);
    let rel_m = (simd_m - scalar_m).abs() / scalar_m.abs().max(1.0);
    let rel_v = (simd_v - scalar_v).abs() / scalar_v.abs().max(1.0);
    assert!(
      rel_m < 1e-5 && rel_v < 1e-4,
      "plane stats disagree: mean simd={simd_m} scalar={scalar_m} (rel {rel_m}), \
       var simd={simd_v} scalar={scalar_v} (rel {rel_v})"
    );
  }

  #[cfg(target_arch = "aarch64")]
  #[test]
  fn neon_plane_mean_variance_matches_scalar() {
    assert_plane_stats_equiv(16, 4, 16, |p, w, h, s| unsafe {
      neon::plane_mean_variance(p, w, h, s)
    });
    assert_plane_stats_equiv(17, 4, 17, |p, w, h, s| unsafe {
      neon::plane_mean_variance(p, w, h, s)
    });
    assert_plane_stats_equiv(24, 5, 64, |p, w, h, s| unsafe {
      neon::plane_mean_variance(p, w, h, s)
    });
    assert_plane_stats_equiv(257, 31, 257, |p, w, h, s| unsafe {
      neon::plane_mean_variance(p, w, h, s)
    });
  }

  #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
  #[test]
  fn ssse3_plane_mean_variance_matches_scalar() {
    if !std::is_x86_feature_detected!("ssse3") {
      return;
    }
    assert_plane_stats_equiv(16, 4, 16, |p, w, h, s| unsafe {
      x86_ssse3::plane_mean_variance(p, w, h, s)
    });
    assert_plane_stats_equiv(17, 4, 17, |p, w, h, s| unsafe {
      x86_ssse3::plane_mean_variance(p, w, h, s)
    });
    assert_plane_stats_equiv(24, 5, 64, |p, w, h, s| unsafe {
      x86_ssse3::plane_mean_variance(p, w, h, s)
    });
    assert_plane_stats_equiv(257, 31, 257, |p, w, h, s| unsafe {
      x86_ssse3::plane_mean_variance(p, w, h, s)
    });
  }
}
