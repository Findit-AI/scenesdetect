//! Platform-specific SIMD (plus a scalar fallback) for the content
//! detector's BGR→HSV conversion.
//!
//! Dispatch is compile-time via `target_arch` — no runtime feature
//! detection is needed because the current SIMD backend (aarch64 NEON)
//! is in every aarch64 target's base ISA. Additional platforms can be
//! added as sibling private modules (e.g. an `x86_ssse3` module exposing
//! its own `bgr_to_hsv_planes`), wired into [`bgr_to_hsv_planes`] via
//! another `cfg` branch.
//!
//! The module is private to `crate::content` — callers in `content.rs`
//! use just the two entry points here; they never see platform details.

// Platform-specific modules, each exposing `pub(super) unsafe fn
// bgr_to_hsv_planes(...)`. Gated so each file is only compiled on matching
// targets — the source need not exist for other arches.

#[cfg(target_arch = "aarch64")]
mod neon;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86_ssse3;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86_avx2;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
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
pub(super) fn bgr_to_hsv_planes(
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  src: &[u8],
  width: u32,
  height: u32,
  stride: u32,
) {
  #[cfg(target_arch = "aarch64")]
  {
    // SAFETY: NEON is part of the base ARMv8-A ISA — every aarch64 Rust
    // target has it. No runtime feature detection required.
    unsafe {
      neon::bgr_to_hsv_planes(h_out, s_out, v_out, src, width, height, stride);
    }
    return;
  }

  #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
  {
    // SAFETY: simd128 target feature enabled at compile time.
    unsafe {
      wasm_simd128::bgr_to_hsv_planes(h_out, s_out, v_out, src, width, height, stride);
    }
    return;
  }

  // x86 runtime dispatch when std is available.
  #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
  {
    if std::is_x86_feature_detected!("avx2") {
      // SAFETY: runtime-checked above.
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
      let h8 = (hue * 0.5).round().clamp(0.0, 179.0) as u8;
      (
        h8,
        s.round().clamp(0.0, 255.0) as u8,
        v.round().clamp(0.0, 255.0) as u8,
      )
    }
  }
}
