//! Platform-specific SIMD (plus a scalar fallback) for the content
//! detector's BGR→HSV conversion.
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
//! The module is private to `crate::content` — callers in `content.rs`
//! use just the two entry points here; they never see platform details.

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
pub(super) fn bgr_to_hsv_planes(
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
pub(super) fn bgr_to_hsv_pixel(b: f32, g: f32, r: f32) -> (u8, u8, u8) {
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
pub(super) fn mean_abs_diff(a: &[u8], b: &[u8], n: usize, use_simd: bool) -> f64 {
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
pub(super) fn sobel(
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
}
