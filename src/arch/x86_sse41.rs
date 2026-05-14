//! x86 / x86_64 SSE4.1 backends for the keyframe-metric SIMD
//! kernels (`noise`, `colorfulness`, `gradient_anisotropy`).
//!
//! Sits between AVX2 and SSSE3 in the runtime dispatch ladder.
//! Algorithmically equivalent to the SSSE3 backends — the
//! SSE4.1-specific changes are:
//!
//! - **`_mm_cvtepu8_epi16`** replaces the SSSE3 idiom
//!   `_mm_unpacklo_epi8(v, zero)` for zero-extending the low 8
//!   bytes of a `__m128i` to `i16×8`. One intrinsic, no zero
//!   register needed.
//! - **`_mm_cvtepu8_epi32`** lets `gradient_anisotropy` drop a
//!   16-byte `pshufb` shuffle table entirely — the 4 dir bytes
//!   are zero-extended straight into `i32×4` lanes by the
//!   intrinsic.
//!
//! Gated on the `sse4.1` target feature. The dispatcher picks
//! this backend only when `is_x86_feature_detected!("sse4.1")`
//! (or `target_feature = "sse4.1"` at compile time).

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// Same PSHUFB deinterleave tables as the SSSE3 backend; SSE4.1
// implies SSSE3 so `_mm_shuffle_epi8` is available here. Kept
// private to this module — duplicating 9 small constants is
// cheaper than reaching across modules.

const BLK0_B: [i8; 16] = [0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
const BLK0_G: [i8; 16] = [1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
const BLK0_R: [i8; 16] = [2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
const BLK1_B: [i8; 16] = [-1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14, -1, -1, -1, -1, -1];
const BLK1_G: [i8; 16] = [-1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1];
const BLK1_R: [i8; 16] = [-1, -1, -1, -1, -1, 1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1];
const BLK2_B: [i8; 16] = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 4, 7, 10, 13];
const BLK2_G: [i8; 16] = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14];
const BLK2_R: [i8; 16] = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15];

/// SSE4.1 Immerkaer noise estimator on a u8 luma plane.
///
/// Algorithm matches the SSSE3 backend exactly; the only
/// SSE4.1 change is `_mm_cvtepu8_epi16` for u8→i16 widening
/// (replaces `_mm_unpacklo_epi8(v, zero)` and removes the need
/// for a zero register).
///
/// # Safety
///
/// Caller must ensure SSE4.1 is available.
#[target_feature(enable = "sse4.1")]
#[allow(unused_unsafe)]
pub(super) unsafe fn noise(luma: &[u8], w: usize, h: usize, s: usize) -> f32 {
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
  let ones16 = unsafe { _mm_set1_epi16(1) };
  let mut acc = unsafe { _mm_setzero_si128() }; // i64×2
  let mut tail_acc: i64 = 0;

  for y in 1..h - 1 {
    let prev = &luma[(y - 1) * s..];
    let curr = &luma[y * s..];
    let next = &luma[(y + 1) * s..];

    let mut x = 1;
    while x < x_vec_end {
      let load8 = |p: *const u8| -> __m128i { unsafe { _mm_loadl_epi64(p as *const __m128i) } };

      let tl = load8(unsafe { prev.as_ptr().add(x - 1) });
      let t = load8(unsafe { prev.as_ptr().add(x) });
      let tr = load8(unsafe { prev.as_ptr().add(x + 1) });
      let l = load8(unsafe { curr.as_ptr().add(x - 1) });
      let c = load8(unsafe { curr.as_ptr().add(x) });
      let r = load8(unsafe { curr.as_ptr().add(x + 1) });
      let bl = load8(unsafe { next.as_ptr().add(x - 1) });
      let b = load8(unsafe { next.as_ptr().add(x) });
      let br = load8(unsafe { next.as_ptr().add(x + 1) });

      // SSE4.1: single-intrinsic u8×8 → i16×8 widening.
      let tl = unsafe { _mm_cvtepu8_epi16(tl) };
      let t = unsafe { _mm_cvtepu8_epi16(t) };
      let tr = unsafe { _mm_cvtepu8_epi16(tr) };
      let l = unsafe { _mm_cvtepu8_epi16(l) };
      let c = unsafe { _mm_cvtepu8_epi16(c) };
      let r = unsafe { _mm_cvtepu8_epi16(r) };
      let bl = unsafe { _mm_cvtepu8_epi16(bl) };
      let b = unsafe { _mm_cvtepu8_epi16(b) };
      let br = unsafe { _mm_cvtepu8_epi16(br) };

      let four_c = unsafe { _mm_slli_epi16::<2>(c) };
      let tblr = unsafe { _mm_add_epi16(_mm_add_epi16(t, b), _mm_add_epi16(l, r)) };
      let two_tblr = unsafe { _mm_slli_epi16::<1>(tblr) };
      let corners = unsafe { _mm_add_epi16(_mm_add_epi16(tl, tr), _mm_add_epi16(bl, br)) };
      let lap = unsafe { _mm_add_epi16(_mm_sub_epi16(four_c, two_tblr), corners) };
      let abs_lap = unsafe { _mm_abs_epi16(lap) };

      let i32_pairs = unsafe { _mm_madd_epi16(abs_lap, ones16) };
      let sum64_a = unsafe { _mm_unpacklo_epi32(i32_pairs, zero) };
      let sum64_b = unsafe { _mm_unpackhi_epi32(i32_pairs, zero) };
      acc = unsafe { _mm_add_epi64(acc, sum64_a) };
      acc = unsafe { _mm_add_epi64(acc, sum64_b) };

      x += LANES;
    }

    while x < w - 1 {
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
      let lap = 4 * c - 2 * (t + b + l + r) + (tl + tr + bl + br);
      tail_acc += lap.unsigned_abs() as i64;
      x += 1;
    }
  }

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

  const COEFF: f64 = 0.208_898_754_886_372_3;
  (((vec_sum + tail_acc) as f64) * COEFF / (interior as f64)) as f32
}

/// SSE4.1 Hasler-Süßstrunk colourfulness on packed 24-bit BGR.
///
/// Algorithm matches the SSSE3 backend. SSE4.1 changes:
/// `_mm_cvtepu8_epi16(v)` replaces `_mm_unpacklo_epi8(v, zero)`
/// for the low 8-lane widening. The high half stays on
/// `_mm_unpackhi_epi8` because `cvtepu8_epi16` only addresses
/// the low 8 bytes.
///
/// Accumulators are `i64×2` (signed sums) / non-negative `i64×2`
/// (squared sums) so no frame size can overflow them.
///
/// # Safety
///
/// Caller must ensure SSE4.1 is available.
#[target_feature(enable = "sse4.1")]
#[allow(unused_unsafe)]
pub(super) unsafe fn colorfulness(bgr: &[u8], w: usize, h: usize, stride: usize) -> f32 {
  let n = w.saturating_mul(h);
  if n == 0 {
    return 0.0;
  }

  const LANES: usize = 16;
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
  let zero = unsafe { _mm_setzero_si128() };
  let ones16 = unsafe { _mm_set1_epi16(1) };

  let mut sum_rg = unsafe { _mm_setzero_si128() }; // i64×2 signed
  let mut sum_u = unsafe { _mm_setzero_si128() }; // i64×2 signed
  let mut sum_rg_sq = unsafe { _mm_setzero_si128() }; // i64×2 non-negative
  let mut sum_u_sq = unsafe { _mm_setzero_si128() }; // i64×2 non-negative
  let mut tail_sum_rg: i64 = 0;
  let mut tail_sum_u: i64 = 0;
  let mut tail_sum_rg_sq: u64 = 0;
  let mut tail_sum_u_sq: u64 = 0;

  for y in 0..h {
    let row_base = y * stride;

    let mut x = 0;
    while x < whole {
      let p = unsafe { bgr.as_ptr().add(row_base + x * 3) };
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

      // Low 8 lanes via SSE4.1 `_mm_cvtepu8_epi16`; high 8 via
      // SSSE3-style `_mm_unpackhi_epi8` (no `cvtepu8_epi16`
      // variant addresses the high half directly).
      let b_lo = unsafe { _mm_cvtepu8_epi16(b) };
      let g_lo = unsafe { _mm_cvtepu8_epi16(g) };
      let r_lo = unsafe { _mm_cvtepu8_epi16(r) };
      let rg_lo = unsafe { _mm_sub_epi16(r_lo, g_lo) };
      let rpg_lo = unsafe { _mm_add_epi16(r_lo, g_lo) };
      let two_b_lo = unsafe { _mm_slli_epi16::<1>(b_lo) };
      let u_lo = unsafe { _mm_sub_epi16(rpg_lo, two_b_lo) };

      let b_hi = unsafe { _mm_unpackhi_epi8(b, zero) };
      let g_hi = unsafe { _mm_unpackhi_epi8(g, zero) };
      let r_hi = unsafe { _mm_unpackhi_epi8(r, zero) };
      let rg_hi = unsafe { _mm_sub_epi16(r_hi, g_hi) };
      let rpg_hi = unsafe { _mm_add_epi16(r_hi, g_hi) };
      let two_b_hi = unsafe { _mm_slli_epi16::<1>(b_hi) };
      let u_hi = unsafe { _mm_sub_epi16(rpg_hi, two_b_hi) };

      // Σ rg / Σ u: pair-sum via madd-with-ones, add halves,
      // sign-extend i32×4 → i64×2 × 2, accumulate.
      let rg_pairs_lo = unsafe { _mm_madd_epi16(rg_lo, ones16) };
      let rg_pairs_hi = unsafe { _mm_madd_epi16(rg_hi, ones16) };
      let rg_pairs = unsafe { _mm_add_epi32(rg_pairs_lo, rg_pairs_hi) };
      let rg_sign = unsafe { _mm_srai_epi32::<31>(rg_pairs) };
      sum_rg = unsafe { _mm_add_epi64(sum_rg, _mm_unpacklo_epi32(rg_pairs, rg_sign)) };
      sum_rg = unsafe { _mm_add_epi64(sum_rg, _mm_unpackhi_epi32(rg_pairs, rg_sign)) };
      let u_pairs_lo = unsafe { _mm_madd_epi16(u_lo, ones16) };
      let u_pairs_hi = unsafe { _mm_madd_epi16(u_hi, ones16) };
      let u_pairs = unsafe { _mm_add_epi32(u_pairs_lo, u_pairs_hi) };
      let u_sign = unsafe { _mm_srai_epi32::<31>(u_pairs) };
      sum_u = unsafe { _mm_add_epi64(sum_u, _mm_unpacklo_epi32(u_pairs, u_sign)) };
      sum_u = unsafe { _mm_add_epi64(sum_u, _mm_unpackhi_epi32(u_pairs, u_sign)) };

      // Σ rg² / Σ u² via madd-with-self.
      let rg_sq_lo = unsafe { _mm_madd_epi16(rg_lo, rg_lo) };
      let rg_sq_hi = unsafe { _mm_madd_epi16(rg_hi, rg_hi) };
      let rg_sq_i32 = unsafe { _mm_add_epi32(rg_sq_lo, rg_sq_hi) };
      let rg_sq_lo64 = unsafe { _mm_unpacklo_epi32(rg_sq_i32, zero) };
      let rg_sq_hi64 = unsafe { _mm_unpackhi_epi32(rg_sq_i32, zero) };
      sum_rg_sq = unsafe { _mm_add_epi64(sum_rg_sq, rg_sq_lo64) };
      sum_rg_sq = unsafe { _mm_add_epi64(sum_rg_sq, rg_sq_hi64) };

      let u_sq_lo = unsafe { _mm_madd_epi16(u_lo, u_lo) };
      let u_sq_hi = unsafe { _mm_madd_epi16(u_hi, u_hi) };
      let u_sq_i32 = unsafe { _mm_add_epi32(u_sq_lo, u_sq_hi) };
      let u_sq_lo64 = unsafe { _mm_unpacklo_epi32(u_sq_i32, zero) };
      let u_sq_hi64 = unsafe { _mm_unpackhi_epi32(u_sq_i32, zero) };
      sum_u_sq = unsafe { _mm_add_epi64(sum_u_sq, u_sq_lo64) };
      sum_u_sq = unsafe { _mm_add_epi64(sum_u_sq, u_sq_hi64) };

      x += LANES;
    }

    while x < w {
      let b = bgr[row_base + x * 3] as i32;
      let g = bgr[row_base + x * 3 + 1] as i32;
      let r = bgr[row_base + x * 3 + 2] as i32;
      let rg = r - g;
      let u = r + g - 2 * b;
      tail_sum_rg += rg as i64;
      tail_sum_u += u as i64;
      tail_sum_rg_sq += (rg * rg) as u64;
      tail_sum_u_sq += (u * u) as u64;
      x += 1;
    }
  }

  let reduce_signed = |v: __m128i| -> i64 {
    let mut lanes = [0i64; 2];
    unsafe { _mm_storeu_si128(lanes.as_mut_ptr() as *mut __m128i, v) };
    lanes[0].wrapping_add(lanes[1])
  };
  let reduce_unsigned = |v: __m128i| -> u64 {
    let mut lanes = [0i64; 2];
    unsafe { _mm_storeu_si128(lanes.as_mut_ptr() as *mut __m128i, v) };
    (lanes[0] as u64).wrapping_add(lanes[1] as u64)
  };

  let total_sum_rg = reduce_signed(sum_rg).wrapping_add(tail_sum_rg);
  let total_sum_u = reduce_signed(sum_u).wrapping_add(tail_sum_u);
  let total_sum_rg_sq = reduce_unsigned(sum_rg_sq).wrapping_add(tail_sum_rg_sq);
  let total_sum_u_sq = reduce_unsigned(sum_u_sq).wrapping_add(tail_sum_u_sq);

  let n_f = n as f64;
  let mean_rg = (total_sum_rg as f64) / n_f;
  let mean_u = (total_sum_u as f64) / n_f;
  let mean_yb = mean_u * 0.5;
  let var_rg = ((total_sum_rg_sq as f64) / n_f - mean_rg * mean_rg).max(0.0);
  let var_u = ((total_sum_u_sq as f64) / n_f - mean_u * mean_u).max(0.0);
  let var_yb = var_u * 0.25;

  let sigma_rgyb = crate::sqrt_64(var_rg + var_yb);
  let mu_rgyb = crate::sqrt_64(mean_rg * mean_rg + mean_yb * mean_yb);
  (sigma_rgyb + 0.3 * mu_rgyb) as f32
}

/// SSE4.1 magnitude-weighted gradient-direction anisotropy.
///
/// The genuine SSE4.1 win over SSSE3:
/// `_mm_cvtepu8_epi32(dir4_v)` zero-extends 4 bytes of `dir`
/// directly into `i32×4` lanes — replaces the SSSE3 idiom
/// (pshufb against a 16-byte mask) and lets us drop the
/// `zext_shuf` constant entirely.
///
/// Rest of the algorithm matches the SSSE3 backend exactly:
/// 4-pixel chunks, four parallel bin streams masked by
/// `_mm_cmpeq_epi32`, `_mm_cmpgt_epi32(mag, 0)` for the
/// positive-mag predicate, `i64×2` per-bin accumulators.
///
/// # Safety
///
/// Caller must ensure SSE4.1 is available.
#[target_feature(enable = "sse4.1")]
#[allow(unused_unsafe)]
pub(super) unsafe fn gradient_anisotropy(mag: &[i32], dir: &[u8], w: usize, h: usize) -> f32 {
  if w < 3 || h < 3 {
    return 0.0;
  }

  const LANES: usize = 4;
  let x_vec_end = if w >= 2 + LANES {
    1 + ((w - 2) / LANES) * LANES
  } else {
    1
  };

  let zero = unsafe { _mm_setzero_si128() };
  let mask3 = unsafe { _mm_set1_epi32(0b11) };
  let bin_consts = [
    unsafe { _mm_setzero_si128() },
    unsafe { _mm_set1_epi32(1) },
    unsafe { _mm_set1_epi32(2) },
    unsafe { _mm_set1_epi32(3) },
  ];

  let mut acc: [__m128i; 4] = [unsafe { _mm_setzero_si128() }; 4];
  let mut tail: [u64; 4] = [0; 4];

  for y in 1..h - 1 {
    let row_off = y * w;

    let mut x = 1;
    while x < x_vec_end {
      let idx = row_off + x;
      let mag4 = unsafe { _mm_loadu_si128(mag.as_ptr().add(idx) as *const __m128i) };

      // SSE4.1: zero-extend 4 dir bytes directly to i32×4. The
      // SSSE3 backend needed a 16-byte `zext_shuf` table and a
      // `pshufb` — none of that is required here.
      let dir4_raw = unsafe { (dir.as_ptr().add(idx) as *const u32).read_unaligned() };
      let dir4_v = unsafe { _mm_cvtsi32_si128(dir4_raw as i32) };
      let dir_i32 = unsafe { _mm_cvtepu8_epi32(dir4_v) };
      let bins_v = unsafe { _mm_and_si128(dir_i32, mask3) };

      let pos_mask = unsafe { _mm_cmpgt_epi32(mag4, zero) };
      let pos_mag = unsafe { _mm_and_si128(mag4, pos_mask) };

      for bin_val in 0..4usize {
        let bin_eq = unsafe { _mm_cmpeq_epi32(bins_v, bin_consts[bin_val]) };
        let masked = unsafe { _mm_and_si128(pos_mag, bin_eq) };
        let lo64 = unsafe { _mm_unpacklo_epi32(masked, zero) };
        let hi64 = unsafe { _mm_unpackhi_epi32(masked, zero) };
        acc[bin_val] = unsafe { _mm_add_epi64(acc[bin_val], lo64) };
        acc[bin_val] = unsafe { _mm_add_epi64(acc[bin_val], hi64) };
      }

      x += LANES;
    }

    while x < w - 1 {
      let idx = row_off + x;
      let m = mag[idx];
      if m > 0 {
        let d = dir[idx] as usize & 0b11;
        tail[d] = tail[d].saturating_add(m as u64);
      }
      x += 1;
    }
  }

  let mut hist = tail;
  for bin_val in 0..4 {
    let mut lanes = [0i64; 2];
    unsafe { _mm_storeu_si128(lanes.as_mut_ptr() as *mut __m128i, acc[bin_val]) };
    hist[bin_val] = hist[bin_val]
      .wrapping_add(lanes[0] as u64)
      .wrapping_add(lanes[1] as u64);
  }

  let total: u64 = hist.iter().sum();
  if total == 0 {
    return 0.0;
  }
  let max_bin = *hist.iter().max().expect("4 bins") as f64;
  let total_f = total as f64;
  let frac = max_bin / total_f;
  ((frac - 0.25).max(0.0) / 0.75) as f32
}
