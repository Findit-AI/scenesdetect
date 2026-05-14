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

use crate::frame::ChannelOrder;

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
#[target_feature(enable = "sse4.1", enable = "ssse3")]
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
#[target_feature(enable = "sse4.1", enable = "ssse3")]
#[allow(unused_unsafe)]
pub(super) unsafe fn colorfulness(
  bgr: &[u8],
  w: usize,
  h: usize,
  stride: usize,
  order: ChannelOrder,
) -> f32 {
  let n = w.saturating_mul(h);
  if n == 0 {
    return 0.0;
  }

  const LANES: usize = 16;
  let whole = w / LANES * LANES;

  let (m_b0, m_b1, m_b2, m_r0, m_r1, m_r2) = match order {
    ChannelOrder::Bgr => unsafe {
      (
        _mm_loadu_si128(BLK0_B.as_ptr() as *const __m128i),
        _mm_loadu_si128(BLK1_B.as_ptr() as *const __m128i),
        _mm_loadu_si128(BLK2_B.as_ptr() as *const __m128i),
        _mm_loadu_si128(BLK0_R.as_ptr() as *const __m128i),
        _mm_loadu_si128(BLK1_R.as_ptr() as *const __m128i),
        _mm_loadu_si128(BLK2_R.as_ptr() as *const __m128i),
      )
    },
    ChannelOrder::Rgb => unsafe {
      (
        _mm_loadu_si128(BLK0_R.as_ptr() as *const __m128i),
        _mm_loadu_si128(BLK1_R.as_ptr() as *const __m128i),
        _mm_loadu_si128(BLK2_R.as_ptr() as *const __m128i),
        _mm_loadu_si128(BLK0_B.as_ptr() as *const __m128i),
        _mm_loadu_si128(BLK1_B.as_ptr() as *const __m128i),
        _mm_loadu_si128(BLK2_B.as_ptr() as *const __m128i),
      )
    },
  };
  let m_g0 = unsafe { _mm_loadu_si128(BLK0_G.as_ptr() as *const __m128i) };
  let m_g1 = unsafe { _mm_loadu_si128(BLK1_G.as_ptr() as *const __m128i) };
  let m_g2 = unsafe { _mm_loadu_si128(BLK2_G.as_ptr() as *const __m128i) };
  let zero = unsafe { _mm_setzero_si128() };
  let ones16 = unsafe { _mm_set1_epi16(1) };

  let (b_off, r_off) = match order {
    ChannelOrder::Bgr => (0usize, 2usize),
    ChannelOrder::Rgb => (2usize, 0usize),
  };

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
      let b = bgr[row_base + x * 3 + b_off] as i32;
      let g = bgr[row_base + x * 3 + 1] as i32;
      let r = bgr[row_base + x * 3 + r_off] as i32;
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
#[target_feature(enable = "sse4.1", enable = "ssse3")]
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

// ============================================================================
// Older kernels: SSE4.1 versions of the SSSE3 backends.
//
// For most of these the only SSE4.1 win is `_mm_cvtepu8_epi16(v)` replacing
// `_mm_unpacklo_epi8(v, zero)` (one intrinsic, no zero register). `mean_abs_diff`
// and `clipping_count` use no SSE4.1-specific intrinsics — their SSE4.1 tier
// exists only so callers running on SSE4.1-capable CPUs get a single
// runtime-feature-detect cost regardless of which kernel they invoke (the
// dispatcher picks the SSE4.1 path for all of them in one go).
//
// `bgr_to_hsv_planes` does use real SSE4.1 features: `_mm_blendv_ps` for the
// per-lane conditional selects (replaces the SSE2 `(mask & t) | (!mask & f)`
// idiom). Rounding stays on the SSSE3/AVX2 add-0.5-then-truncate idiom —
// SSE4.1's `_mm_round_ps::<NEAREST_INT>` is ties-to-even and would drift
// by 1 LSB on half-value inputs vs the scalar `round()` which is
// ties-away-from-zero.
// ============================================================================

/// SSE4.1 `mean_abs_diff`: identical to the SSSE3 backend (no SSE4.1
/// intrinsics that meaningfully change `_mm_sad_epu8` + i64x2 reduce).
///
/// # Safety
///
/// Caller must ensure SSE4.1 is available.
#[target_feature(enable = "sse4.1", enable = "ssse3")]
#[allow(unused_unsafe)]
pub(super) unsafe fn mean_abs_diff(a: &[u8], b: &[u8], n: usize) -> f64 {
  const LANES: usize = 16;
  let whole = n / LANES * LANES;
  let mut acc = unsafe { _mm_setzero_si128() };

  let mut i = 0;
  while i < whole {
    let va = unsafe { _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i) };
    let vb = unsafe { _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i) };
    let sad = unsafe { _mm_sad_epu8(va, vb) };
    acc = unsafe { _mm_add_epi64(acc, sad) };
    i += LANES;
  }

  let hi = unsafe { _mm_srli_si128::<8>(acc) };
  let total = unsafe { _mm_add_epi64(acc, hi) };
  #[cfg(target_arch = "x86_64")]
  let mut sum: u64 = unsafe { _mm_cvtsi128_si64(total) as u64 };
  #[cfg(target_arch = "x86")]
  let mut sum: u64 = {
    let mut tmp = 0u64;
    unsafe { _mm_storel_epi64(&mut tmp as *mut u64 as *mut __m128i, total) };
    tmp
  };

  while i < n {
    let da = a[i] as i32 - b[i] as i32;
    sum += da.unsigned_abs() as u64;
    i += 1;
  }

  sum as f64 / n as f64
}

/// SSE4.1 Sobel 3×3. Identical structure to the SSSE3 backend, with
/// `_mm_cvtepu8_epi16` replacing `_mm_unpacklo_epi8(v, zero)` for u8→i16
/// widening and `_mm_cvtepi16_epi32` replacing the
/// `_mm_unpacklo_epi16(v, _mm_cmpgt_epi16(zero, v))` sign-extend idiom for
/// i16→i32 widening (one intrinsic instead of three).
///
/// # Safety
///
/// Caller must ensure SSE4.1 is available.
#[target_feature(enable = "sse4.1", enable = "ssse3")]
#[allow(unused_unsafe)]
pub(super) unsafe fn sobel(input: &[u8], mag: &mut [i32], dir: &mut [u8], w: usize, h: usize) {
  mag.fill(0);
  dir.fill(0);

  const LANES: usize = 8;

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
          unsafe { _mm_cvtepu8_epi16(v) }
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

      let gx = unsafe {
        let pos = _mm_add_epi16(_mm_add_epi16(pr, _mm_slli_epi16::<1>(cr)), nr);
        let neg = _mm_add_epi16(_mm_add_epi16(pl, _mm_slli_epi16::<1>(cl)), nl);
        _mm_sub_epi16(pos, neg)
      };
      let gy = unsafe {
        let pos = _mm_add_epi16(_mm_add_epi16(nl, _mm_slli_epi16::<1>(nm)), nr);
        let neg = _mm_add_epi16(_mm_add_epi16(pl, _mm_slli_epi16::<1>(pm)), pr);
        _mm_sub_epi16(pos, neg)
      };

      let mag_i16 = unsafe { _mm_add_epi16(_mm_abs_epi16(gx), _mm_abs_epi16(gy)) };

      // SSE4.1 sign-extending widening i16x8 → two i32x4. The high half
      // needs a shift-down first because `cvtepi16_epi32` only addresses
      // the low 4 lanes.
      let lo = unsafe { _mm_cvtepi16_epi32(mag_i16) };
      let hi = unsafe { _mm_cvtepi16_epi32(_mm_srli_si128::<8>(mag_i16)) };
      unsafe {
        _mm_storeu_si128(mag.as_mut_ptr().add(off + x) as *mut __m128i, lo);
        _mm_storeu_si128(mag.as_mut_ptr().add(off + x + 4) as *mut __m128i, hi);
      }

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

/// SSE4.1 BGR → BT.601 luma. Same deinterleave + coefficient-MAC pattern as
/// the SSSE3 backend; `_mm_cvtepu8_epi16` replaces `_mm_unpacklo_epi8(v, zero)`
/// for the low half. The high half stays on `_mm_unpackhi_epi8`.
///
/// # Safety
///
/// Caller must ensure SSE4.1 is available.
#[target_feature(enable = "sse4.1", enable = "ssse3")]
#[allow(unused_unsafe)]
pub(super) unsafe fn bgr_to_luma(
  out: &mut [u8],
  src: &[u8],
  width: u32,
  height: u32,
  stride: u32,
  order: ChannelOrder,
) {
  const LANES: usize = 16;
  let w = width as usize;
  let h = height as usize;
  let s = stride as usize;
  let whole = w / LANES * LANES;

  let (m_b0, m_b1, m_b2, m_r0, m_r1, m_r2) = match order {
    ChannelOrder::Bgr => unsafe {
      (
        _mm_loadu_si128(BLK0_B.as_ptr() as *const __m128i),
        _mm_loadu_si128(BLK1_B.as_ptr() as *const __m128i),
        _mm_loadu_si128(BLK2_B.as_ptr() as *const __m128i),
        _mm_loadu_si128(BLK0_R.as_ptr() as *const __m128i),
        _mm_loadu_si128(BLK1_R.as_ptr() as *const __m128i),
        _mm_loadu_si128(BLK2_R.as_ptr() as *const __m128i),
      )
    },
    ChannelOrder::Rgb => unsafe {
      (
        _mm_loadu_si128(BLK0_R.as_ptr() as *const __m128i),
        _mm_loadu_si128(BLK1_R.as_ptr() as *const __m128i),
        _mm_loadu_si128(BLK2_R.as_ptr() as *const __m128i),
        _mm_loadu_si128(BLK0_B.as_ptr() as *const __m128i),
        _mm_loadu_si128(BLK1_B.as_ptr() as *const __m128i),
        _mm_loadu_si128(BLK2_B.as_ptr() as *const __m128i),
      )
    },
  };
  let m_g0 = unsafe { _mm_loadu_si128(BLK0_G.as_ptr() as *const __m128i) };
  let m_g1 = unsafe { _mm_loadu_si128(BLK1_G.as_ptr() as *const __m128i) };
  let m_g2 = unsafe { _mm_loadu_si128(BLK2_G.as_ptr() as *const __m128i) };
  let zero = unsafe { _mm_setzero_si128() };

  let (b_off, r_off) = match order {
    ChannelOrder::Bgr => (0usize, 2usize),
    ChannelOrder::Rgb => (2usize, 0usize),
  };

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

      // SSE4.1: single-intrinsic u8×8 → i16×8 widening for the low half.
      let b_lo = unsafe { _mm_cvtepu8_epi16(b) };
      let g_lo = unsafe { _mm_cvtepu8_epi16(g) };
      let r_lo = unsafe { _mm_cvtepu8_epi16(r) };
      let acc_lo = unsafe {
        _mm_add_epi16(
          _mm_add_epi16(_mm_mullo_epi16(b_lo, k_b), _mm_mullo_epi16(g_lo, k_g)),
          _mm_mullo_epi16(r_lo, k_r),
        )
      };

      // High half: cvtepu8_epi16 addresses the low 8 bytes, so use the
      // SSSE3 unpackhi idiom for the upper 8.
      let b_hi = unsafe { _mm_unpackhi_epi8(b, zero) };
      let g_hi = unsafe { _mm_unpackhi_epi8(g, zero) };
      let r_hi = unsafe { _mm_unpackhi_epi8(r, zero) };
      let acc_hi = unsafe {
        _mm_add_epi16(
          _mm_add_epi16(_mm_mullo_epi16(b_hi, k_b), _mm_mullo_epi16(g_hi, k_g)),
          _mm_mullo_epi16(r_hi, k_r),
        )
      };

      let y_lo = unsafe { _mm_srli_epi16(acc_lo, 8) };
      let y_hi = unsafe { _mm_srli_epi16(acc_hi, 8) };
      let packed = unsafe { _mm_packus_epi16(y_lo, y_hi) };
      unsafe {
        _mm_storeu_si128(out.as_mut_ptr().add(dst_off + x) as *mut __m128i, packed);
      }

      x += LANES;
    }

    while x < w {
      let b = src[row_base + x * 3 + b_off] as u32;
      let g = src[row_base + x * 3 + 1] as u32;
      let r = src[row_base + x * 3 + r_off] as u32;
      out[dst_off + x] = ((77 * r + 150 * g + 29 * b) >> 8) as u8;
      x += 1;
    }
  }
}

/// SSE4.1 clipping-pixel count. Same algorithm as SSSE3 (`max_epu8` +
/// `subs_epu8` flags + PSADBW reduction); SSE4.1 has no useful intrinsics
/// for this kernel, so the body is identical apart from `target_feature`.
///
/// # Safety
///
/// Caller must ensure SSE4.1 is available.
#[target_feature(enable = "sse4.1", enable = "ssse3")]
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
      let under = unsafe { _mm_subs_epu8(lo, max) };
      let over = unsafe { _mm_subs_epu8(max, hi) };
      let flags = unsafe { _mm_or_si128(under, over) };
      let flags_01 = unsafe { _mm_min_epu8(flags, one) };
      let sad = unsafe { _mm_sad_epu8(flags_01, zero) };
      acc = unsafe { _mm_add_epi64(acc, sad) };

      x += LANES;
    }

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

/// SSE4.1 Tenengrad. Identical to SSSE3 except for `_mm_cvtepu8_epi16`
/// replacing the unpacklo_epi8-against-zero widening.
///
/// # Safety
///
/// Caller must ensure SSE4.1 is available.
#[target_feature(enable = "sse4.1", enable = "ssse3")]
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
      let load8 = |p: *const u8| -> __m128i { unsafe { _mm_loadl_epi64(p as *const __m128i) } };

      let tl = load8(unsafe { prev.as_ptr().add(x - 1) });
      let t = load8(unsafe { prev.as_ptr().add(x) });
      let tr = load8(unsafe { prev.as_ptr().add(x + 1) });
      let l = load8(unsafe { curr.as_ptr().add(x - 1) });
      let r = load8(unsafe { curr.as_ptr().add(x + 1) });
      let bl = load8(unsafe { next.as_ptr().add(x - 1) });
      let b = load8(unsafe { next.as_ptr().add(x) });
      let br = load8(unsafe { next.as_ptr().add(x + 1) });

      // SSE4.1 widening.
      let tl = unsafe { _mm_cvtepu8_epi16(tl) };
      let t = unsafe { _mm_cvtepu8_epi16(t) };
      let tr = unsafe { _mm_cvtepu8_epi16(tr) };
      let l = unsafe { _mm_cvtepu8_epi16(l) };
      let r = unsafe { _mm_cvtepu8_epi16(r) };
      let bl = unsafe { _mm_cvtepu8_epi16(bl) };
      let b = unsafe { _mm_cvtepu8_epi16(b) };
      let br = unsafe { _mm_cvtepu8_epi16(br) };

      let two_l = unsafe { _mm_slli_epi16::<1>(l) };
      let two_r = unsafe { _mm_slli_epi16::<1>(r) };
      let pos_x = unsafe { _mm_add_epi16(_mm_add_epi16(tr, two_r), br) };
      let neg_x = unsafe { _mm_add_epi16(_mm_add_epi16(tl, two_l), bl) };
      let gx = unsafe { _mm_sub_epi16(pos_x, neg_x) };

      let two_t = unsafe { _mm_slli_epi16::<1>(t) };
      let two_b = unsafe { _mm_slli_epi16::<1>(b) };
      let pos_y = unsafe { _mm_add_epi16(_mm_add_epi16(bl, two_b), br) };
      let neg_y = unsafe { _mm_add_epi16(_mm_add_epi16(tl, two_t), tr) };
      let gy = unsafe { _mm_sub_epi16(pos_y, neg_y) };

      let lo_pair = unsafe { _mm_unpacklo_epi16(gx, gy) };
      let hi_pair = unsafe { _mm_unpackhi_epi16(gx, gy) };
      let sq_lo = unsafe { _mm_madd_epi16(lo_pair, lo_pair) };
      let sq_hi = unsafe { _mm_madd_epi16(hi_pair, hi_pair) };

      let sum32 = unsafe { _mm_add_epi32(sq_lo, sq_hi) };

      let sum64_a = unsafe { _mm_unpacklo_epi32(sum32, zero) };
      let sum64_b = unsafe { _mm_unpackhi_epi32(sum32, zero) };
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

/// SSE4.1 single-pass `(mean, variance)` on a u8 plane. Same SAD-based
/// sum and `madd_epi16(v, v)` squared sum as SSSE3, with
/// `_mm_cvtepu8_epi16` swapped in for the low-half widening.
///
/// # Safety
///
/// Caller must ensure SSE4.1 is available.
#[target_feature(enable = "sse4.1", enable = "ssse3")]
#[allow(unused_unsafe)]
pub(super) unsafe fn plane_mean_variance(plane: &[u8], w: usize, h: usize, s: usize) -> (f32, f32) {
  const LANES: usize = 16;
  let n = w.saturating_mul(h);
  if n == 0 {
    return (0.0, 0.0);
  }
  let whole = w / LANES * LANES;

  let zero = unsafe { _mm_setzero_si128() };
  let mut sum_acc = unsafe { _mm_setzero_si128() }; // u64x2
  let mut sq_acc = unsafe { _mm_setzero_si128() }; // u64x2
  let mut tail_sum: u64 = 0;
  let mut tail_sq: u64 = 0;

  for y in 0..h {
    let row_base = y * s;

    let mut x = 0;
    while x < whole {
      let v = unsafe { _mm_loadu_si128(plane.as_ptr().add(row_base + x) as *const __m128i) };

      let sad = unsafe { _mm_sad_epu8(v, zero) };
      sum_acc = unsafe { _mm_add_epi64(sum_acc, sad) };

      // SSE4.1 widening for the low half; SSSE3 unpackhi for high.
      let v_lo = unsafe { _mm_cvtepu8_epi16(v) };
      let v_hi = unsafe { _mm_unpackhi_epi8(v, zero) };
      let pair_lo = unsafe { _mm_madd_epi16(v_lo, v_lo) };
      let pair_hi = unsafe { _mm_madd_epi16(v_hi, v_hi) };
      let pair_sum = unsafe { _mm_add_epi32(pair_lo, pair_hi) };
      let part_a = unsafe { _mm_unpacklo_epi32(pair_sum, zero) };
      let part_b = unsafe { _mm_unpackhi_epi32(pair_sum, zero) };
      sq_acc = unsafe { _mm_add_epi64(sq_acc, part_a) };
      sq_acc = unsafe { _mm_add_epi64(sq_acc, part_b) };

      x += LANES;
    }

    while x < w {
      let v = plane[row_base + x] as u64;
      tail_sum += v;
      tail_sq += v * v;
      x += 1;
    }
  }

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
  let sq = sq_vec + tail_sq;
  let n_f = n as f64;
  let mean = sum as f64 / n_f;
  let var = (sq as f64 / n_f - mean * mean).max(0.0);
  (mean as f32, var as f32)
}

/// SSE4.1 BGR→HSV. Uses SSE4.1's `_mm_blendv_ps` (true per-lane select,
/// replaces the SSE2 `(mask & t) | (!mask & f)` idiom — saves one op per
/// blend). Rounding stays on the add-0.5-then-truncate SSSE3/AVX2 idiom
/// (NOT `_mm_round_ps::<NEAREST_INT>`, which is ties-to-even and would
/// drift by 1 LSB on half-value inputs vs the scalar `round()`'s
/// ties-away-from-zero). The channel widening uses `_mm_cvtepu8_epi16`
/// for the low half and
/// `_mm_unpackhi_epi8(v, zero)` for the high half, mirroring the
/// `bgr_to_luma` pattern above.
///
/// # Safety
///
/// Caller must ensure SSE4.1 is available.
#[target_feature(enable = "sse4.1", enable = "ssse3")]
#[allow(unused_unsafe)]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn bgr_to_hsv_planes(
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  src: &[u8],
  width: u32,
  height: u32,
  stride: u32,
  order: ChannelOrder,
) {
  const LANES: usize = 16;
  let w = width as usize;
  let h = height as usize;
  let s = stride as usize;
  let whole = w / LANES * LANES;

  let (m_b0, m_b1, m_b2, m_r0, m_r1, m_r2) = match order {
    ChannelOrder::Bgr => unsafe {
      (
        _mm_loadu_si128(BLK0_B.as_ptr() as *const __m128i),
        _mm_loadu_si128(BLK1_B.as_ptr() as *const __m128i),
        _mm_loadu_si128(BLK2_B.as_ptr() as *const __m128i),
        _mm_loadu_si128(BLK0_R.as_ptr() as *const __m128i),
        _mm_loadu_si128(BLK1_R.as_ptr() as *const __m128i),
        _mm_loadu_si128(BLK2_R.as_ptr() as *const __m128i),
      )
    },
    ChannelOrder::Rgb => unsafe {
      (
        _mm_loadu_si128(BLK0_R.as_ptr() as *const __m128i),
        _mm_loadu_si128(BLK1_R.as_ptr() as *const __m128i),
        _mm_loadu_si128(BLK2_R.as_ptr() as *const __m128i),
        _mm_loadu_si128(BLK0_B.as_ptr() as *const __m128i),
        _mm_loadu_si128(BLK1_B.as_ptr() as *const __m128i),
        _mm_loadu_si128(BLK2_B.as_ptr() as *const __m128i),
      )
    },
  };
  let m_g0 = unsafe { _mm_loadu_si128(BLK0_G.as_ptr() as *const __m128i) };
  let m_g1 = unsafe { _mm_loadu_si128(BLK1_G.as_ptr() as *const __m128i) };
  let m_g2 = unsafe { _mm_loadu_si128(BLK2_G.as_ptr() as *const __m128i) };
  let zero_i = unsafe { _mm_setzero_si128() };

  let (b_off, r_off) = match order {
    ChannelOrder::Bgr => (0usize, 2usize),
    ChannelOrder::Rgb => (2usize, 0usize),
  };

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

      // SSE4.1 widening for low halves; SSSE3 unpackhi for high.
      let b_lo16 = unsafe { _mm_cvtepu8_epi16(b) };
      let b_hi16 = unsafe { _mm_unpackhi_epi8(b, zero_i) };
      let g_lo16 = unsafe { _mm_cvtepu8_epi16(g) };
      let g_hi16 = unsafe { _mm_unpackhi_epi8(g, zero_i) };
      let r_lo16 = unsafe { _mm_cvtepu8_epi16(r) };
      let r_hi16 = unsafe { _mm_unpackhi_epi8(r, zero_i) };

      macro_rules! group {
        ($b16:expr, $g16:expr, $r16:expr, $half:ident) => {{
          let bu = unsafe { $half($b16, zero_i) };
          let gu = unsafe { $half($g16, zero_i) };
          let ru = unsafe { $half($r16, zero_i) };
          let bf = unsafe { _mm_cvtepi32_ps(bu) };
          let gf = unsafe { _mm_cvtepi32_ps(gu) };
          let rf = unsafe { _mm_cvtepi32_ps(ru) };
          let (hue, sat, val) = unsafe { bgr_to_hsv_f32x4_sse41(bf, gf, rf) };
          // Scalar uses Rust `f32::round()` which is ties-away-from-zero;
          // for our non-negative inputs that's the add-0.5-then-truncate
          // pattern. SSE4.1's `_mm_round_ps::<NEAREST_INT>` is ties-to-
          // even — half-values like BGR(5,6,6)'s saturation = 42.5 would
          // round to 42 instead of the scalar's 43, producing 1-LSB drift
          // on SSE4.1-only x86 hosts. Match the SSSE3 / AVX2 pattern
          // (`_mm_cvttps_epi32(_mm_add_ps(v, 0.5))`) for bit-identical
          // results across every backend.
          let half = unsafe { _mm_set1_ps(0.5) };
          let hh = unsafe { _mm_mul_ps(hue, half) };
          let h_u32 = unsafe { clamp_i32_max_sse41(_mm_cvttps_epi32(_mm_add_ps(hh, half)), 179) };
          let s_u32 = unsafe { clamp_i32_max_sse41(_mm_cvttps_epi32(_mm_add_ps(sat, half)), 255) };
          let v_u32 = unsafe { clamp_i32_max_sse41(_mm_cvttps_epi32(_mm_add_ps(val, half)), 255) };
          (h_u32, s_u32, v_u32)
        }};
      }

      let (h0, s0, v0) = group!(b_lo16, g_lo16, r_lo16, _mm_unpacklo_epi16);
      let (h1, s1, v1) = group!(b_lo16, g_lo16, r_lo16, _mm_unpackhi_epi16);
      let (h2, s2, v2) = group!(b_hi16, g_hi16, r_hi16, _mm_unpacklo_epi16);
      let (h3, s3, v3) = group!(b_hi16, g_hi16, r_hi16, _mm_unpackhi_epi16);

      let h_vec = unsafe { pack_quad_sse41(h0, h1, h2, h3) };
      let s_vec = unsafe { pack_quad_sse41(s0, s1, s2, s3) };
      let v_vec = unsafe { pack_quad_sse41(v0, v1, v2, v3) };

      unsafe {
        _mm_storeu_si128(h_out.as_mut_ptr().add(dst_off + x) as *mut __m128i, h_vec);
        _mm_storeu_si128(s_out.as_mut_ptr().add(dst_off + x) as *mut __m128i, s_vec);
        _mm_storeu_si128(v_out.as_mut_ptr().add(dst_off + x) as *mut __m128i, v_vec);
      }

      x += LANES;
    }

    // Scalar tail — single-pixel loop matches the SSSE3 backend.
    let row = &src[row_base..row_base + w * 3];
    while x < w {
      let b = row[x * 3 + b_off] as f32;
      let g = row[x * 3 + 1] as f32;
      let r = row[x * 3 + r_off] as f32;
      let (hue, sat, val) = super::scalar::Scalar::bgr_to_hsv_pixel(b, g, r);
      h_out[dst_off + x] = hue;
      s_out[dst_off + x] = sat;
      v_out[dst_off + x] = val;
      x += 1;
    }
  }
}

/// SSE4.1 `_mm_min_epi32`-based clamp. SSE4.1 adds packed-signed-i32
/// min/max, removing the need for the SSE2 compare+blend idiom.
#[target_feature(enable = "sse4.1", enable = "ssse3")]
#[allow(unused_unsafe)]
#[inline]
unsafe fn clamp_i32_max_sse41(v: __m128i, max: i32) -> __m128i {
  unsafe { _mm_min_epi32(v, _mm_set1_epi32(max)) }
}

/// SSE4.1 packing helper (`_mm_packus_epi32` is SSE4.1; the SSSE3 path
/// had to use `_mm_packs_epi32` because PACKUSDW didn't exist below
/// SSE4.1).
#[target_feature(enable = "sse4.1", enable = "ssse3")]
#[allow(unused_unsafe)]
#[inline]
unsafe fn pack_quad_sse41(a: __m128i, b: __m128i, c: __m128i, d: __m128i) -> __m128i {
  // PACKUSDW: i32×4 + i32×4 → u16×8 with unsigned saturation.
  let lo = unsafe { _mm_packus_epi32(a, b) };
  let hi = unsafe { _mm_packus_epi32(c, d) };
  unsafe { _mm_packus_epi16(lo, hi) }
}

/// SSE4.1 branch-free 4-lane BGR→HSV core. Uses `_mm_blendv_ps` for the
/// per-lane selects (single instruction, replaces the
/// `(mask & t) | (!mask & f)` SSE2 idiom).
#[target_feature(enable = "sse4.1", enable = "ssse3")]
#[allow(unused_unsafe)]
#[inline]
unsafe fn bgr_to_hsv_f32x4_sse41(b: __m128, g: __m128, r: __m128) -> (__m128, __m128, __m128) {
  let zero = unsafe { _mm_setzero_ps() };
  let one = unsafe { _mm_set1_ps(1.0) };

  let v = unsafe { _mm_max_ps(_mm_max_ps(b, g), r) };
  let min = unsafe { _mm_min_ps(_mm_min_ps(b, g), r) };
  let delta = unsafe { _mm_sub_ps(v, min) };

  let delta_zero = unsafe { _mm_cmpeq_ps(delta, zero) };
  let v_zero = unsafe { _mm_cmpeq_ps(v, zero) };
  // `_mm_blendv_ps(f, t, mask)` selects t-lane when mask's sign bit is 1.
  let delta_safe = unsafe { _mm_blendv_ps(delta, one, delta_zero) };

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
  let hue_rg = unsafe { _mm_blendv_ps(h_b, h_r, is_r) };
  let hue = unsafe { _mm_blendv_ps(hue_rg, h_g, not_r_and_g) };
  let neg = unsafe { _mm_cmplt_ps(hue, zero) };
  let hue = unsafe { _mm_blendv_ps(hue, _mm_add_ps(hue, c360), neg) };
  let hue = unsafe { _mm_blendv_ps(hue, zero, delta_zero) };

  let v_safe = unsafe { _mm_blendv_ps(v, one, v_zero) };
  let sat = unsafe { _mm_div_ps(_mm_mul_ps(c255, delta), v_safe) };
  let sat = unsafe { _mm_blendv_ps(sat, zero, v_zero) };

  (hue, sat, v)
}
