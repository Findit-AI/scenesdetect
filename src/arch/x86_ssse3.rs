//! x86 / x86_64 SSSE3 backend for BGR→HSV.
//!
//! No native 3-channel deinterleave on x86; we emulate it with `PSHUFB`
//! (SSSE3). Nine shuffle masks + six ORs deinterleave 48 packed BGR bytes
//! into three `u8x16` vectors. The rest of the pipeline mirrors the NEON
//! version: widen u8→u16→u32, convert to f32x4, run the branch-free HSV
//! math on four 4-pixel groups, narrow back to u8x16 via saturating packs.
//!
//! SSE4.1's `_mm_blendv_ps` would be nicer for mask blending but we stick to
//! SSSE3 + SSE2 (universal on x86_64). The manual `(mask & t) | (!mask & f)`
//! pattern compiles to the same handful of ops.

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// Shuffle masks for PSHUFB (`_mm_shuffle_epi8`). Each mask has one byte per
// output lane: if high bit is set, output lane is zeroed; else low 4 bits
// select the input byte. We use `-1` for "zero this lane".
//
// Input blocks (16 bytes each):
//   blk0: B0 G0 R0 B1 G1 R1 B2 G2 R2 B3 G3 R3 B4 G4 R4 B5
//   blk1: G5 R5 B6 G6 R6 B7 G7 R7 B8 G8 R8 B9 G9 R9 B10 G10
//   blk2: R10 B11 G11 R11 B12 G12 R12 B13 G13 R13 B14 G14 R14 B15 G15 R15

// When AVX2 is also enabled at compile time, the BGR→HSV dispatch takes
// the AVX2 path, leaving the SSSE3 BGR function + its helpers and shuffle
// constants unused. `mean_abs_diff` and `sobel` are still called via SSSE3
// even when AVX2 is present (no AVX2 variants of those exist).
#[allow(dead_code)]
const BLK0_B: [i8; 16] = [0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
#[allow(dead_code)]
const BLK0_G: [i8; 16] = [1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
#[allow(dead_code)]
const BLK0_R: [i8; 16] = [2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];

#[allow(dead_code)]
const BLK1_B: [i8; 16] = [-1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14, -1, -1, -1, -1, -1];
#[allow(dead_code)]
const BLK1_G: [i8; 16] = [-1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1];
#[allow(dead_code)]
const BLK1_R: [i8; 16] = [-1, -1, -1, -1, -1, 1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1];

#[allow(dead_code)]
const BLK2_B: [i8; 16] = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 4, 7, 10, 13];
#[allow(dead_code)]
const BLK2_G: [i8; 16] = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14];
#[allow(dead_code)]
const BLK2_R: [i8; 16] = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15];

/// SSSE3 BGR→HSV: 16 pixels per iteration.
///
/// # Safety
///
/// Caller must ensure SSSE3 is available (`is_x86_feature_detected!("ssse3")`
/// or `target_feature = "ssse3"`). Buffers must cover the ranges indicated by
/// `width`, `height`, `stride`.
#[allow(dead_code)] // AVX2 takes the BGR path when both are compiled
#[target_feature(enable = "ssse3")]
#[allow(unused_unsafe)]
pub(super) unsafe fn bgr_to_hsv_planes(
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  src: &[u8],
  width: u32,
  height: u32,
  stride: u32,
) {
  const LANES: usize = 16;
  let w = width as usize;
  let h = height as usize;
  let s = stride as usize;
  let whole = w / LANES * LANES;

  let m_b0 = unsafe { _mm_loadu_si128(BLK0_B.as_ptr() as *const __m128i) };
  let m_g0 = unsafe { _mm_loadu_si128(BLK0_G.as_ptr() as *const __m128i) };
  let m_r0 = unsafe { _mm_loadu_si128(BLK0_R.as_ptr() as *const __m128i) };
  let m_b1 = unsafe { _mm_loadu_si128(BLK1_B.as_ptr() as *const __m128i) };
  let m_g1 = unsafe { _mm_loadu_si128(BLK1_G.as_ptr() as *const __m128i) };
  let m_r1 = unsafe { _mm_loadu_si128(BLK1_R.as_ptr() as *const __m128i) };
  let m_b2 = unsafe { _mm_loadu_si128(BLK2_B.as_ptr() as *const __m128i) };
  let m_g2 = unsafe { _mm_loadu_si128(BLK2_G.as_ptr() as *const __m128i) };
  let m_r2 = unsafe { _mm_loadu_si128(BLK2_R.as_ptr() as *const __m128i) };
  let zero_i = unsafe { _mm_setzero_si128() };

  for y in 0..h {
    let row_base = y * s;
    let dst_off = y * w;

    let mut x = 0;
    while x < whole {
      let p = unsafe { src.as_ptr().add(row_base + x * 3) };
      let blk0 = unsafe { _mm_loadu_si128(p as *const __m128i) };
      let blk1 = unsafe { _mm_loadu_si128(p.add(16) as *const __m128i) };
      let blk2 = unsafe { _mm_loadu_si128(p.add(32) as *const __m128i) };

      let b = unsafe {
        _mm_or_si128(
          _mm_or_si128(_mm_shuffle_epi8(blk0, m_b0), _mm_shuffle_epi8(blk1, m_b1)),
          _mm_shuffle_epi8(blk2, m_b2),
        )
      };
      let g = unsafe {
        _mm_or_si128(
          _mm_or_si128(_mm_shuffle_epi8(blk0, m_g0), _mm_shuffle_epi8(blk1, m_g1)),
          _mm_shuffle_epi8(blk2, m_g2),
        )
      };
      let r = unsafe {
        _mm_or_si128(
          _mm_or_si128(_mm_shuffle_epi8(blk0, m_r0), _mm_shuffle_epi8(blk1, m_r1)),
          _mm_shuffle_epi8(blk2, m_r2),
        )
      };

      // Widen u8x16 → two u16x8 halves per channel.
      let b_lo16 = unsafe { _mm_unpacklo_epi8(b, zero_i) };
      let b_hi16 = unsafe { _mm_unpackhi_epi8(b, zero_i) };
      let g_lo16 = unsafe { _mm_unpacklo_epi8(g, zero_i) };
      let g_hi16 = unsafe { _mm_unpackhi_epi8(g, zero_i) };
      let r_lo16 = unsafe { _mm_unpacklo_epi8(r, zero_i) };
      let r_hi16 = unsafe { _mm_unpackhi_epi8(r, zero_i) };

      // Process four groups of 4 pixels each.
      macro_rules! group {
        ($b16:expr, $g16:expr, $r16:expr, $half:ident) => {{
          let bu = unsafe { $half($b16, zero_i) };
          let gu = unsafe { $half($g16, zero_i) };
          let ru = unsafe { $half($r16, zero_i) };
          let bf = unsafe { _mm_cvtepi32_ps(bu) };
          let gf = unsafe { _mm_cvtepi32_ps(gu) };
          let rf = unsafe { _mm_cvtepi32_ps(ru) };
          let (hue, sat, val) = unsafe { bgr_to_hsv_f32x4(bf, gf, rf) };
          // Use add-0.5 + truncate (round half-up for non-negative values)
          // to match the scalar `round()` semantics instead of MXCSR's
          // default round-to-nearest-even via `_mm_cvtps_epi32`.
          let half = unsafe { _mm_set1_ps(0.5) };
          let hh = unsafe { _mm_mul_ps(hue, _mm_set1_ps(0.5)) };
          let h_u32 = unsafe { clamp_i32_max(_mm_cvttps_epi32(_mm_add_ps(hh, half)), 179) };
          let s_u32 = unsafe { clamp_i32_max(_mm_cvttps_epi32(_mm_add_ps(sat, half)), 255) };
          let v_u32 = unsafe { clamp_i32_max(_mm_cvttps_epi32(_mm_add_ps(val, half)), 255) };
          (h_u32, s_u32, v_u32)
        }};
      }

      let (h0, s0, v0) = group!(b_lo16, g_lo16, r_lo16, _mm_unpacklo_epi16);
      let (h1, s1, v1) = group!(b_lo16, g_lo16, r_lo16, _mm_unpackhi_epi16);
      let (h2, s2, v2) = group!(b_hi16, g_hi16, r_hi16, _mm_unpacklo_epi16);
      let (h3, s3, v3) = group!(b_hi16, g_hi16, r_hi16, _mm_unpackhi_epi16);

      let h_vec = unsafe { pack_quad(h0, h1, h2, h3) };
      let s_vec = unsafe { pack_quad(s0, s1, s2, s3) };
      let v_vec = unsafe { pack_quad(v0, v1, v2, v3) };

      unsafe {
        _mm_storeu_si128(h_out.as_mut_ptr().add(dst_off + x) as *mut __m128i, h_vec);
        _mm_storeu_si128(s_out.as_mut_ptr().add(dst_off + x) as *mut __m128i, s_vec);
        _mm_storeu_si128(v_out.as_mut_ptr().add(dst_off + x) as *mut __m128i, v_vec);
      }

      x += LANES;
    }

    // Scalar tail.
    let row = &src[row_base..row_base + w * 3];
    while x < w {
      let b = row[x * 3] as f32;
      let g = row[x * 3 + 1] as f32;
      let r = row[x * 3 + 2] as f32;
      let (hue, sat, val) = super::scalar::Scalar::bgr_to_hsv_pixel(b, g, r);
      h_out[dst_off + x] = hue;
      s_out[dst_off + x] = sat;
      v_out[dst_off + x] = val;
      x += 1;
    }
  }
}

/// Clamp `i32x4` lanes to `[0, max]`. Our values are non-negative by
/// construction (widened from `u8`), so no lower-bound check needed.
#[allow(dead_code)]
#[target_feature(enable = "ssse3")]
#[allow(unused_unsafe)]
#[inline]
unsafe fn clamp_i32_max(v: __m128i, max: i32) -> __m128i {
  let mv = unsafe { _mm_set1_epi32(max) };
  let gt = unsafe { _mm_cmpgt_epi32(v, mv) };
  unsafe { _mm_or_si128(_mm_and_si128(gt, mv), _mm_andnot_si128(gt, v)) }
}

/// Pack four `i32x4` vectors (values ≤ 255) into one `u8x16` via two levels
/// of saturating narrow.
#[allow(dead_code)]
#[target_feature(enable = "ssse3")]
#[allow(unused_unsafe)]
#[inline]
unsafe fn pack_quad(a: __m128i, b: __m128i, c: __m128i, d: __m128i) -> __m128i {
  // _mm_packs_epi32: signed saturation to i16 range (values 0..255 OK).
  let lo = unsafe { _mm_packs_epi32(a, b) };
  let hi = unsafe { _mm_packs_epi32(c, d) };
  // _mm_packus_epi16: unsigned saturation to u8 range.
  unsafe { _mm_packus_epi16(lo, hi) }
}

/// Branch-free 4-lane BGR→HSV core. Returns `(hue ∈ [0, 360), sat, val)` as
/// `f32x4`. Caller divides hue by 2, rounds, and narrows to u8.
#[allow(dead_code)]
#[target_feature(enable = "ssse3")]
#[allow(unused_unsafe)]
#[inline]
unsafe fn bgr_to_hsv_f32x4(b: __m128, g: __m128, r: __m128) -> (__m128, __m128, __m128) {
  let zero = unsafe { _mm_setzero_ps() };
  let one = unsafe { _mm_set1_ps(1.0) };

  let v = unsafe { _mm_max_ps(_mm_max_ps(b, g), r) };
  let min = unsafe { _mm_min_ps(_mm_min_ps(b, g), r) };
  let delta = unsafe { _mm_sub_ps(v, min) };

  let delta_zero = unsafe { _mm_cmpeq_ps(delta, zero) };
  let v_zero = unsafe { _mm_cmpeq_ps(v, zero) };
  let delta_safe = unsafe { blend(delta_zero, one, delta) };

  let sixty = unsafe { _mm_set1_ps(60.0) };
  let c120 = unsafe { _mm_set1_ps(120.0) };
  let c240 = unsafe { _mm_set1_ps(240.0) };
  let c360 = unsafe { _mm_set1_ps(360.0) };
  let c255 = unsafe { _mm_set1_ps(255.0) };

  let h_r = unsafe { _mm_div_ps(_mm_mul_ps(sixty, _mm_sub_ps(g, b)), delta_safe) };
  let h_g = unsafe {
    _mm_add_ps(
      _mm_div_ps(_mm_mul_ps(sixty, _mm_sub_ps(b, r)), delta_safe),
      c120,
    )
  };
  let h_b = unsafe {
    _mm_add_ps(
      _mm_div_ps(_mm_mul_ps(sixty, _mm_sub_ps(r, g)), delta_safe),
      c240,
    )
  };

  let is_r = unsafe { _mm_cmpeq_ps(v, r) };
  let is_g = unsafe { _mm_cmpeq_ps(v, g) };
  let not_r_and_g = unsafe { _mm_andnot_ps(is_r, is_g) };
  let hue_rg = unsafe { blend(is_r, h_r, h_b) };
  let hue = unsafe { blend(not_r_and_g, h_g, hue_rg) };
  let neg = unsafe { _mm_cmplt_ps(hue, zero) };
  let hue = unsafe { blend(neg, _mm_add_ps(hue, c360), hue) };
  let hue = unsafe { blend(delta_zero, zero, hue) };

  let v_safe = unsafe { blend(v_zero, one, v) };
  let sat = unsafe { _mm_div_ps(_mm_mul_ps(c255, delta), v_safe) };
  let sat = unsafe { blend(v_zero, zero, sat) };

  (hue, sat, v)
}

/// `mask ? t : f`, where `mask` is per-lane all-ones or all-zeros from a
/// comparison intrinsic. SSE2 equivalent of SSE4.1 `_mm_blendv_ps`.
#[allow(dead_code)]
#[target_feature(enable = "ssse3")]
#[allow(unused_unsafe)]
#[inline]
unsafe fn blend(mask: __m128, t: __m128, f: __m128) -> __m128 {
  unsafe { _mm_or_ps(_mm_and_ps(mask, t), _mm_andnot_ps(mask, f)) }
}

/// SSE2 `mean_abs_diff`: `Σ|a[i] - b[i]| / n`.
///
/// Uses `_mm_sad_epu8` — a single instruction that computes the sum of
/// absolute u8 differences for 16 bytes, returning two u16 partial sums
/// in lanes 0 and 8 of a `__m128i` (the other lanes are zero).
///
/// # Safety
///
/// Caller must ensure at least SSE2 is available (true on every x86_64 target).
/// Marked `ssse3` because the parent module is ssse3-gated, but only SSE2
/// instructions are used here.
#[target_feature(enable = "ssse3")]
#[allow(unused_unsafe)]
pub(super) unsafe fn mean_abs_diff(a: &[u8], b: &[u8], n: usize) -> f64 {
  const LANES: usize = 16;
  let whole = n / LANES * LANES;
  let mut acc = unsafe { _mm_setzero_si128() }; // u64x2 accumulator

  let mut i = 0;
  while i < whole {
    let va = unsafe { _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i) };
    let vb = unsafe { _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i) };
    // _mm_sad_epu8: per 8-byte half, sums |a[j]-b[j]| into a u16 in
    // lanes 0 and 8. The other 6 lanes of each half are zero.
    let sad = unsafe { _mm_sad_epu8(va, vb) };
    acc = unsafe { _mm_add_epi64(acc, sad) };
    i += LANES;
  }

  // Horizontal reduce u64x2 → u64.
  let hi = unsafe { _mm_srli_si128::<8>(acc) };
  let total = unsafe { _mm_add_epi64(acc, hi) };
  // `_mm_cvtsi128_si64` is x86_64-only (no 64-bit GPRs on i686).
  // Fall back to a memory round-trip on 32-bit.
  #[cfg(target_arch = "x86_64")]
  let mut sum: u64 = unsafe { _mm_cvtsi128_si64(total) as u64 };
  #[cfg(target_arch = "x86")]
  let mut sum: u64 = {
    let mut tmp = 0u64;
    unsafe { _mm_storel_epi64(&mut tmp as *mut u64 as *mut __m128i, total) };
    tmp
  };

  // Scalar tail.
  while i < n {
    let da = a[i] as i32 - b[i] as i32;
    sum += da.unsigned_abs() as u64;
    i += 1;
  }

  sum as f64 / n as f64
}

/// SSSE3 Sobel 3×3. Same structure as NEON: i16x8 stencil for magnitude,
/// scalar direction.
///
/// # Safety
///
/// Caller must ensure SSSE3 is available.
#[target_feature(enable = "ssse3")]
#[allow(unused_unsafe)]
pub(super) unsafe fn sobel(input: &[u8], mag: &mut [i32], dir: &mut [u8], w: usize, h: usize) {
  mag.fill(0);
  dir.fill(0);

  const LANES: usize = 8;
  let zero_i = unsafe { _mm_setzero_si128() };

  for y in 1..h.saturating_sub(1) {
    let prev = &input[(y - 1) * w..];
    let curr = &input[y * w..];
    let next = &input[(y + 1) * w..];
    let off = y * w;

    let mut x = 1usize;

    while x + LANES < w {
      macro_rules! ld {
        ($row:expr, $o:expr) => {{
          let v = unsafe { _mm_loadl_epi64($row.as_ptr().add($o) as *const __m128i) };
          unsafe { _mm_unpacklo_epi8(v, zero_i) } // u8→u16, treated as i16 (values 0..255)
        }};
      }
      let pl = ld!(prev, x - 1);
      let pm = ld!(prev, x);
      let pr = ld!(prev, x + 1);
      let cl = ld!(curr, x - 1);
      let cr = ld!(curr, x + 1);
      let nl = ld!(next, x - 1);
      let nm = ld!(next, x);
      let nr = ld!(next, x + 1);

      // Gx = (pr + 2*cr + nr) - (pl + 2*cl + nl)
      let gx = unsafe {
        let pos = _mm_add_epi16(_mm_add_epi16(pr, _mm_slli_epi16::<1>(cr)), nr);
        let neg = _mm_add_epi16(_mm_add_epi16(pl, _mm_slli_epi16::<1>(cl)), nl);
        _mm_sub_epi16(pos, neg)
      };
      // Gy = (nl + 2*nm + nr) - (pl + 2*pm + pr)
      let gy = unsafe {
        let pos = _mm_add_epi16(_mm_add_epi16(nl, _mm_slli_epi16::<1>(nm)), nr);
        let neg = _mm_add_epi16(_mm_add_epi16(pl, _mm_slli_epi16::<1>(pm)), pr);
        _mm_sub_epi16(pos, neg)
      };

      let mag_i16 = unsafe { _mm_add_epi16(_mm_abs_epi16(gx), _mm_abs_epi16(gy)) };

      // Widen i16→i32 and store.
      let lo = unsafe { _mm_unpacklo_epi16(mag_i16, _mm_cmpgt_epi16(zero_i, mag_i16)) };
      let hi = unsafe { _mm_unpackhi_epi16(mag_i16, _mm_cmpgt_epi16(zero_i, mag_i16)) };
      unsafe {
        _mm_storeu_si128(mag.as_mut_ptr().add(off + x) as *mut __m128i, lo);
        _mm_storeu_si128(mag.as_mut_ptr().add(off + x + 4) as *mut __m128i, hi);
      }

      // Direction: scalar.
      let gx_arr: [i16; 8] = unsafe { core::mem::transmute(gx) };
      let gy_arr: [i16; 8] = unsafe { core::mem::transmute(gy) };
      for j in 0..LANES {
        let ax = gx_arr[j].unsigned_abs() as u32;
        let ay = gy_arr[j].unsigned_abs() as u32;
        dir[off + x + j] = if ay * 1000 < ax * 414 {
          0
        } else if ay * 1000 > ax * 2414 {
          2
        } else if (gx_arr[j] >= 0) == (gy_arr[j] >= 0) {
          1
        } else {
          3
        };
      }

      x += LANES;
    }

    // Scalar tail.
    while x < w - 1 {
      let i = |yy: usize, xx: usize| input[yy * w + xx] as i32;
      let gx = -i(y - 1, x - 1) - 2 * i(y, x - 1) - i(y + 1, x - 1)
        + i(y - 1, x + 1)
        + 2 * i(y, x + 1)
        + i(y + 1, x + 1);
      let gy = -i(y - 1, x - 1) - 2 * i(y - 1, x) - i(y - 1, x + 1)
        + i(y + 1, x - 1)
        + 2 * i(y + 1, x)
        + i(y + 1, x + 1);
      mag[off + x] = gx.abs() + gy.abs();
      let ax = gx.unsigned_abs();
      let ay = gy.unsigned_abs();
      dir[off + x] = if ay * 1000 < ax * 414 {
        0
      } else if ay * 1000 > ax * 2414 {
        2
      } else if gx.signum() == gy.signum() {
        1
      } else {
        3
      };
      x += 1;
    }
  }
}

/// SSSE3 BGR → BT.601 luma: `Y = (77·R + 150·G + 29·B) >> 8`.
///
/// Same 9-mask deinterleave as [`bgr_to_hsv_planes`] produces `B`, `G`,
/// `R` as three `u8x16` vectors; we then widen each to two `u16x8`
/// halves, `PMULLW` against splatted coefficients, sum, shift right by
/// 8, and `PACKUSWB` back to `u8x16`. Coefficients sum to 256 so the
/// u16 accumulator tops out at 65280 (no saturation risk). Tail is
/// scalar.
///
/// # Safety
///
/// Caller must ensure SSSE3 is available.
#[target_feature(enable = "ssse3")]
#[allow(unused_unsafe)]
pub(super) unsafe fn bgr_to_luma(out: &mut [u8], src: &[u8], width: u32, height: u32, stride: u32) {
  const LANES: usize = 16;
  let w = width as usize;
  let h = height as usize;
  let s = stride as usize;
  let whole = w / LANES * LANES;

  // Deinterleave masks (reused from bgr_to_hsv_planes).
  let m_b0 = unsafe { _mm_loadu_si128(BLK0_B.as_ptr() as *const __m128i) };
  let m_g0 = unsafe { _mm_loadu_si128(BLK0_G.as_ptr() as *const __m128i) };
  let m_r0 = unsafe { _mm_loadu_si128(BLK0_R.as_ptr() as *const __m128i) };
  let m_b1 = unsafe { _mm_loadu_si128(BLK1_B.as_ptr() as *const __m128i) };
  let m_g1 = unsafe { _mm_loadu_si128(BLK1_G.as_ptr() as *const __m128i) };
  let m_r1 = unsafe { _mm_loadu_si128(BLK1_R.as_ptr() as *const __m128i) };
  let m_b2 = unsafe { _mm_loadu_si128(BLK2_B.as_ptr() as *const __m128i) };
  let m_g2 = unsafe { _mm_loadu_si128(BLK2_G.as_ptr() as *const __m128i) };
  let m_r2 = unsafe { _mm_loadu_si128(BLK2_R.as_ptr() as *const __m128i) };
  let zero = unsafe { _mm_setzero_si128() };

  // Coefficient splats, u16 lanes.
  let k_b = unsafe { _mm_set1_epi16(29) };
  let k_g = unsafe { _mm_set1_epi16(150) };
  let k_r = unsafe { _mm_set1_epi16(77) };

  for y in 0..h {
    let row_base = y * s;
    let dst_off = y * w;

    let mut x = 0;
    while x < whole {
      let p = unsafe { src.as_ptr().add(row_base + x * 3) };
      let blk0 = unsafe { _mm_loadu_si128(p as *const __m128i) };
      let blk1 = unsafe { _mm_loadu_si128(p.add(16) as *const __m128i) };
      let blk2 = unsafe { _mm_loadu_si128(p.add(32) as *const __m128i) };

      let b = unsafe {
        _mm_or_si128(
          _mm_or_si128(_mm_shuffle_epi8(blk0, m_b0), _mm_shuffle_epi8(blk1, m_b1)),
          _mm_shuffle_epi8(blk2, m_b2),
        )
      };
      let g = unsafe {
        _mm_or_si128(
          _mm_or_si128(_mm_shuffle_epi8(blk0, m_g0), _mm_shuffle_epi8(blk1, m_g1)),
          _mm_shuffle_epi8(blk2, m_g2),
        )
      };
      let r = unsafe {
        _mm_or_si128(
          _mm_or_si128(_mm_shuffle_epi8(blk0, m_r0), _mm_shuffle_epi8(blk1, m_r1)),
          _mm_shuffle_epi8(blk2, m_r2),
        )
      };

      // Low 8 lanes: widen to u16 and compute 29·B + 150·G + 77·R.
      let b_lo = unsafe { _mm_unpacklo_epi8(b, zero) };
      let g_lo = unsafe { _mm_unpacklo_epi8(g, zero) };
      let r_lo = unsafe { _mm_unpacklo_epi8(r, zero) };
      let acc_lo = unsafe {
        _mm_add_epi16(
          _mm_add_epi16(_mm_mullo_epi16(b_lo, k_b), _mm_mullo_epi16(g_lo, k_g)),
          _mm_mullo_epi16(r_lo, k_r),
        )
      };

      // High 8 lanes.
      let b_hi = unsafe { _mm_unpackhi_epi8(b, zero) };
      let g_hi = unsafe { _mm_unpackhi_epi8(g, zero) };
      let r_hi = unsafe { _mm_unpackhi_epi8(r, zero) };
      let acc_hi = unsafe {
        _mm_add_epi16(
          _mm_add_epi16(_mm_mullo_epi16(b_hi, k_b), _mm_mullo_epi16(g_hi, k_g)),
          _mm_mullo_epi16(r_hi, k_r),
        )
      };

      // >>8, then pack both halves back to u8×16 (saturation a no-op —
      // values are already in [0, 255]).
      let y_lo = unsafe { _mm_srli_epi16(acc_lo, 8) };
      let y_hi = unsafe { _mm_srli_epi16(acc_hi, 8) };
      let packed = unsafe { _mm_packus_epi16(y_lo, y_hi) };
      unsafe {
        _mm_storeu_si128(out.as_mut_ptr().add(dst_off + x) as *mut __m128i, packed);
      }

      x += LANES;
    }

    // Scalar tail.
    while x < w {
      let b = src[row_base + x * 3] as u32;
      let g = src[row_base + x * 3 + 1] as u32;
      let r = src[row_base + x * 3 + 2] as u32;
      out[dst_off + x] = ((77 * r + 150 * g + 29 * b) >> 8) as u8;
      x += 1;
    }
  }
}

/// SSSE3 clipping-pixel count. 16 pixels per iteration via the same
/// 9-shuffle deinterleave; per-pixel `max(B,G,R)` via `_mm_max_epu8`;
/// two unsigned-saturation subtractions (`_mm_subs_epu8(5, max)` and
/// `_mm_subs_epu8(max, 250)`) produce nonzero bytes exactly where the
/// channel max lies outside `[5, 250]`. `PSADBW` against zero
/// horizontally sums u8 lanes per 64-bit half, giving us the count of
/// (0 or 1) flags per half; extract and accumulate. Tail scalar.
///
/// # Safety
///
/// Caller must ensure SSSE3 is available.
#[target_feature(enable = "ssse3")]
#[allow(unused_unsafe)]
pub(super) unsafe fn clipping_count(src: &[u8], width: u32, height: u32, stride: u32) -> u64 {
  const LANES: usize = 16;
  let w = width as usize;
  let h = height as usize;
  let s = stride as usize;
  let whole = w / LANES * LANES;

  let m_b0 = unsafe { _mm_loadu_si128(BLK0_B.as_ptr() as *const __m128i) };
  let m_g0 = unsafe { _mm_loadu_si128(BLK0_G.as_ptr() as *const __m128i) };
  let m_r0 = unsafe { _mm_loadu_si128(BLK0_R.as_ptr() as *const __m128i) };
  let m_b1 = unsafe { _mm_loadu_si128(BLK1_B.as_ptr() as *const __m128i) };
  let m_g1 = unsafe { _mm_loadu_si128(BLK1_G.as_ptr() as *const __m128i) };
  let m_r1 = unsafe { _mm_loadu_si128(BLK1_R.as_ptr() as *const __m128i) };
  let m_b2 = unsafe { _mm_loadu_si128(BLK2_B.as_ptr() as *const __m128i) };
  let m_g2 = unsafe { _mm_loadu_si128(BLK2_G.as_ptr() as *const __m128i) };
  let m_r2 = unsafe { _mm_loadu_si128(BLK2_R.as_ptr() as *const __m128i) };

  let lo = unsafe { _mm_set1_epi8(5) };
  let hi = unsafe { _mm_set1_epi8(250u8 as i8) };
  let one = unsafe { _mm_set1_epi8(1) };
  let zero = unsafe { _mm_setzero_si128() };

  // u64x2 accumulator — same pattern as mean_abs_diff.
  let mut acc = unsafe { _mm_setzero_si128() };
  let mut scalar_count: u64 = 0;

  for y in 0..h {
    let row_base = y * s;

    let mut x = 0;
    while x < whole {
      let p = unsafe { src.as_ptr().add(row_base + x * 3) };
      let blk0 = unsafe { _mm_loadu_si128(p as *const __m128i) };
      let blk1 = unsafe { _mm_loadu_si128(p.add(16) as *const __m128i) };
      let blk2 = unsafe { _mm_loadu_si128(p.add(32) as *const __m128i) };

      let b = unsafe {
        _mm_or_si128(
          _mm_or_si128(_mm_shuffle_epi8(blk0, m_b0), _mm_shuffle_epi8(blk1, m_b1)),
          _mm_shuffle_epi8(blk2, m_b2),
        )
      };
      let g = unsafe {
        _mm_or_si128(
          _mm_or_si128(_mm_shuffle_epi8(blk0, m_g0), _mm_shuffle_epi8(blk1, m_g1)),
          _mm_shuffle_epi8(blk2, m_g2),
        )
      };
      let r = unsafe {
        _mm_or_si128(
          _mm_or_si128(_mm_shuffle_epi8(blk0, m_r0), _mm_shuffle_epi8(blk1, m_r1)),
          _mm_shuffle_epi8(blk2, m_r2),
        )
      };

      let max = unsafe { _mm_max_epu8(b, _mm_max_epu8(g, r)) };

      // `subs_epu8(5, max)` is 0 when max >= 5, and (5 - max) > 0 else.
      // `subs_epu8(max, 250)` is 0 when max <= 250, and (max - 250) > 0 else.
      let under = unsafe { _mm_subs_epu8(lo, max) };
      let over = unsafe { _mm_subs_epu8(max, hi) };
      let flags = unsafe { _mm_or_si128(under, over) };

      // Clamp any nonzero byte down to 1, zero stays zero. `min(flags, 1)`
      // turns the 0-or-small bytes into a clean 0/1 mask.
      let flags_01 = unsafe { _mm_min_epu8(flags, one) };

      // PSADBW against zero: each 64-bit half of the result holds the
      // sum of the 8 u8 lanes in that half — i.e. the count of 1s.
      let sad = unsafe { _mm_sad_epu8(flags_01, zero) };
      acc = unsafe { _mm_add_epi64(acc, sad) };

      x += LANES;
    }

    // Scalar tail — accumulate separately to avoid extra vector ops.
    while x < w {
      let b = src[row_base + x * 3];
      let g = src[row_base + x * 3 + 1];
      let r = src[row_base + x * 3 + 2];
      let m = b.max(g).max(r);
      if !(5..=250).contains(&m) {
        scalar_count += 1;
      }
      x += 1;
    }
  }

  // Horizontal reduce u64x2 → u64 via shift-and-add, same as mean_abs_diff.
  let hi_half = unsafe { _mm_srli_si128::<8>(acc) };
  let total = unsafe { _mm_add_epi64(acc, hi_half) };
  #[cfg(target_arch = "x86_64")]
  let vec_count: u64 = unsafe { _mm_cvtsi128_si64(total) } as u64;
  #[cfg(target_arch = "x86")]
  let vec_count: u64 = {
    let mut tmp = 0u64;
    unsafe { _mm_storel_epi64(&mut tmp as *mut u64 as *mut __m128i, total) };
    tmp
  };

  vec_count + scalar_count
}

/// SSSE3 Tenengrad: 8 interior pixels per iteration.
///
/// For each 8-pixel slice we load three u8×8 chunks per input row
/// (at offsets `x-1`, `x`, `x+1`) via plain 64-bit loads, widen each
/// to i16×8 with `PUNPCKLBW` against zero, and compute `gx` / `gy`
/// lane-wise with i16 adds / subs / left-shift-by-1.
///
/// Per-pixel `gx² + gy²` comes from `PMADDWD` on interleaved `gx` and
/// `gy` vectors: `madd(interleave(gx, gy), interleave(gx, gy)) = (gx[0]² + gy[0]², …)`
/// as i32×4. Two halves of the 8 pixels are handled via
/// `PUNPCKLWD` / `PUNPCKHWD`.
///
/// The per-iteration i32×4 sum is widened to `i64×2 × 2` with
/// `PUNPCKLDQ` / `PUNPCKHDQ` against zero (safe because squared sums
/// are non-negative) and added into the i64×2 accumulator. At the
/// end, horizontal reduce to a scalar i64 and divide.
///
/// # Safety
///
/// Caller must ensure SSSE3 is available.
#[target_feature(enable = "ssse3")]
#[allow(unused_unsafe)]
pub(super) unsafe fn tenengrad(luma: &[u8], w: usize, h: usize, s: usize) -> f32 {
  if w < 3 || h < 3 {
    return 0.0;
  }
  let interior = (w - 2) * (h - 2);
  if interior == 0 {
    return 0.0;
  }

  const LANES: usize = 8;
  let x_vec_end = if w >= 2 + LANES {
    1 + ((w - 2) / LANES) * LANES
  } else {
    1
  };

  let zero = unsafe { _mm_setzero_si128() };
  let mut acc = unsafe { _mm_setzero_si128() }; // i64x2
  let mut tail_acc: i64 = 0;

  for y in 1..h - 1 {
    let prev = &luma[(y - 1) * s..];
    let curr = &luma[y * s..];
    let next = &luma[(y + 1) * s..];

    let mut x = 1;
    while x < x_vec_end {
      // Load u8×8 chunks (8 bytes), lifted into __m128i with the low 8 bytes valid.
      let load8 = |p: *const u8| -> __m128i { unsafe { _mm_loadl_epi64(p as *const __m128i) } };

      let tl = load8(unsafe { prev.as_ptr().add(x - 1) });
      let t = load8(unsafe { prev.as_ptr().add(x) });
      let tr = load8(unsafe { prev.as_ptr().add(x + 1) });
      let l = load8(unsafe { curr.as_ptr().add(x - 1) });
      let r = load8(unsafe { curr.as_ptr().add(x + 1) });
      let bl = load8(unsafe { next.as_ptr().add(x - 1) });
      let b = load8(unsafe { next.as_ptr().add(x) });
      let br = load8(unsafe { next.as_ptr().add(x + 1) });

      // Zero-extend u8 → i16×8 (values fit, sign doesn't matter since u8 < 128).
      let tl = unsafe { _mm_unpacklo_epi8(tl, zero) };
      let t = unsafe { _mm_unpacklo_epi8(t, zero) };
      let tr = unsafe { _mm_unpacklo_epi8(tr, zero) };
      let l = unsafe { _mm_unpacklo_epi8(l, zero) };
      let r = unsafe { _mm_unpacklo_epi8(r, zero) };
      let bl = unsafe { _mm_unpacklo_epi8(bl, zero) };
      let b = unsafe { _mm_unpacklo_epi8(b, zero) };
      let br = unsafe { _mm_unpacklo_epi8(br, zero) };

      // gx = -tl - 2l - bl + tr + 2r + br
      let two_l = unsafe { _mm_slli_epi16::<1>(l) };
      let two_r = unsafe { _mm_slli_epi16::<1>(r) };
      let pos_x = unsafe { _mm_add_epi16(_mm_add_epi16(tr, two_r), br) };
      let neg_x = unsafe { _mm_add_epi16(_mm_add_epi16(tl, two_l), bl) };
      let gx = unsafe { _mm_sub_epi16(pos_x, neg_x) };

      // gy = -tl - 2t - tr + bl + 2b + br
      let two_t = unsafe { _mm_slli_epi16::<1>(t) };
      let two_b = unsafe { _mm_slli_epi16::<1>(b) };
      let pos_y = unsafe { _mm_add_epi16(_mm_add_epi16(bl, two_b), br) };
      let neg_y = unsafe { _mm_add_epi16(_mm_add_epi16(tl, two_t), tr) };
      let gy = unsafe { _mm_sub_epi16(pos_y, neg_y) };

      // Interleave gx and gy at i16 granularity, then PMADDWD gives
      // per-pair sum of squares.
      let lo_pair = unsafe { _mm_unpacklo_epi16(gx, gy) };
      let hi_pair = unsafe { _mm_unpackhi_epi16(gx, gy) };
      let sq_lo = unsafe { _mm_madd_epi16(lo_pair, lo_pair) }; // i32×4: pixels 0..4
      let sq_hi = unsafe { _mm_madd_epi16(hi_pair, hi_pair) }; // i32×4: pixels 4..8

      // Sum the two i32×4 vectors.
      let sum32 = unsafe { _mm_add_epi32(sq_lo, sq_hi) };

      // Widen i32×4 → two i64×2 by interleaving with zero.
      let sum64_a = unsafe { _mm_unpacklo_epi32(sum32, zero) };
      let sum64_b = unsafe { _mm_unpackhi_epi32(sum32, zero) };
      acc = unsafe { _mm_add_epi64(acc, sum64_a) };
      acc = unsafe { _mm_add_epi64(acc, sum64_b) };

      x += LANES;
    }

    // Scalar tail.
    while x < w - 1 {
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
      tail_acc += (gx * gx + gy * gy) as i64;
      x += 1;
    }
  }

  // Horizontal reduce i64x2 → i64.
  let hi_half = unsafe { _mm_srli_si128::<8>(acc) };
  let total = unsafe { _mm_add_epi64(acc, hi_half) };
  #[cfg(target_arch = "x86_64")]
  let vec_sum: i64 = unsafe { _mm_cvtsi128_si64(total) };
  #[cfg(target_arch = "x86")]
  let vec_sum: i64 = {
    let mut tmp = 0i64;
    unsafe { _mm_storel_epi64(&mut tmp as *mut i64 as *mut __m128i, total) };
    tmp
  };

  (((vec_sum + tail_acc) as f64) / (interior as f64)) as f32
}

/// SSSE3 single-pass `(mean, variance)` on a u8 plane.
///
/// Per 16-byte chunk:
/// - `_mm_sad_epu8(v, zero)` gives two u64 half-sums for `sum_x` —
///   the classic SAD horizontal-sum trick.
/// - Squared sum: widen u8×16 to two u16×8 halves via
///   `_mm_unpacklo/hi_epi8(v, zero)`, then `_mm_mullo_epi16` for
///   per-lane squares (lane max 65025 < 2¹⁶), then `_mm_madd_epi16`
///   against `1` to pair-sum into i32×4 (max 2·65025 = 130050). Sum
///   both halves and add via `_mm_unpacklo/hi_epi32`-widening into a
///   u64x2 squared-sum accumulator.
///
/// # Safety
///
/// Caller must ensure SSSE3 is available.
#[target_feature(enable = "ssse3")]
#[allow(unused_unsafe)]
pub(super) unsafe fn plane_mean_variance(plane: &[u8], w: usize, h: usize, s: usize) -> (f32, f32) {
  const LANES: usize = 16;
  let n = w.saturating_mul(h);
  if n == 0 {
    return (0.0, 0.0);
  }
  let whole = w / LANES * LANES;

  let zero = unsafe { _mm_setzero_si128() };
  let ones_i16 = unsafe { _mm_set1_epi16(1) };
  let mut sum_acc = unsafe { _mm_setzero_si128() }; // u64x2
  let mut sq_acc = unsafe { _mm_setzero_si128() }; // u64x2
  let mut tail_sum: u64 = 0;
  let mut tail_sq: u64 = 0;

  for y in 0..h {
    let row_base = y * s;

    let mut x = 0;
    while x < whole {
      let v = unsafe { _mm_loadu_si128(plane.as_ptr().add(row_base + x) as *const __m128i) };

      // sum_x — PSADBW against zero gives two u64 halves of summed u8.
      let sad = unsafe { _mm_sad_epu8(v, zero) };
      sum_acc = unsafe { _mm_add_epi64(sum_acc, sad) };

      // sum_x²: per-lane u8² → u16, pair-summed via PMADDWD.
      let v_lo = unsafe { _mm_unpacklo_epi8(v, zero) };
      let v_hi = unsafe { _mm_unpackhi_epi8(v, zero) };
      let sq_lo_u16 = unsafe { _mm_mullo_epi16(v_lo, v_lo) };
      let sq_hi_u16 = unsafe { _mm_mullo_epi16(v_hi, v_hi) };
      // madd(v, 1) pair-sums u16×8 → i32×4 (pair max 2·65025 < 2³¹).
      let pair_lo = unsafe { _mm_madd_epi16(sq_lo_u16, ones_i16) };
      let pair_hi = unsafe { _mm_madd_epi16(sq_hi_u16, ones_i16) };
      let pair_sum = unsafe { _mm_add_epi32(pair_lo, pair_hi) };
      // Widen i32×4 → two i64×2 (non-negative, zero-extend).
      let part_a = unsafe { _mm_unpacklo_epi32(pair_sum, zero) };
      let part_b = unsafe { _mm_unpackhi_epi32(pair_sum, zero) };
      sq_acc = unsafe { _mm_add_epi64(sq_acc, part_a) };
      sq_acc = unsafe { _mm_add_epi64(sq_acc, part_b) };

      x += LANES;
    }

    // Scalar tail.
    while x < w {
      let v = plane[row_base + x] as u64;
      tail_sum += v;
      tail_sq += v * v;
      x += 1;
    }
  }

  // Horizontal reduce u64x2 → u64 for both accumulators.
  let sum_hi = unsafe { _mm_srli_si128::<8>(sum_acc) };
  let sum_total = unsafe { _mm_add_epi64(sum_acc, sum_hi) };
  let sq_hi = unsafe { _mm_srli_si128::<8>(sq_acc) };
  let sq_total = unsafe { _mm_add_epi64(sq_acc, sq_hi) };

  #[cfg(target_arch = "x86_64")]
  let (sum_vec, sq_vec) = unsafe {
    (
      _mm_cvtsi128_si64(sum_total) as u64,
      _mm_cvtsi128_si64(sq_total) as u64,
    )
  };
  #[cfg(target_arch = "x86")]
  let (sum_vec, sq_vec) = {
    let mut s = 0u64;
    let mut sq = 0u64;
    unsafe {
      _mm_storel_epi64(&mut s as *mut u64 as *mut __m128i, sum_total);
      _mm_storel_epi64(&mut sq as *mut u64 as *mut __m128i, sq_total);
    }
    (s, sq)
  };

  let sum = sum_vec + tail_sum;
  let sum_sq = sq_vec + tail_sq;
  let n_f = n as f64;
  let mean = (sum as f64) / n_f;
  let mean_sq = (sum_sq as f64) / n_f;
  let variance = (mean_sq - mean * mean).max(0.0);
  (mean as f32, variance as f32)
}
