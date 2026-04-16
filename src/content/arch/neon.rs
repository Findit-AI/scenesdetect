//! Aarch64 NEON backend for BGR→HSV (3-channel deinterleave via `vld3q_u8`).

use core::arch::aarch64::*;

#[target_feature(enable = "neon")]
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

  for y in 0..h {
    let row_base = y * s;
    let dst_off = y * w;

    let mut x = 0;
    while x < whole {
      // Deinterleave 16 BGR pixels (48 bytes) into three u8x16 vectors.
      let bgr = unsafe { vld3q_u8(src.as_ptr().add(row_base + x * 3)) };
      let b = bgr.0;
      let g = bgr.1;
      let r = bgr.2;

      // Per channel: u8x16 → two u16x8 halves.
      let b_lo16 = unsafe { vmovl_u8(vget_low_u8(b)) };
      let b_hi16 = unsafe { vmovl_high_u8(b) };
      let g_lo16 = unsafe { vmovl_u8(vget_low_u8(g)) };
      let g_hi16 = unsafe { vmovl_high_u8(g) };
      let r_lo16 = unsafe { vmovl_u8(vget_low_u8(r)) };
      let r_hi16 = unsafe { vmovl_high_u8(r) };

      // Four 4-pixel groups: {0..4, 4..8, 8..12, 12..16}.
      macro_rules! process_group {
        ($b16:expr, $g16:expr, $r16:expr, $half:ident) => {{
          let bu32 = unsafe { $half($b16) };
          let gu32 = unsafe { $half($g16) };
          let ru32 = unsafe { $half($r16) };
          let bf = unsafe { vcvtq_f32_u32(bu32) };
          let gf = unsafe { vcvtq_f32_u32(gu32) };
          let rf = unsafe { vcvtq_f32_u32(ru32) };
          let (hue, sat, val) = unsafe { bgr_to_hsv_f32x4(bf, gf, rf) };
          // Hue/2 → u32, clamp [0, 179]; S/V → u32, clamp [0, 255].
          let hue_half = unsafe { vmulq_n_f32(hue, 0.5) };
          let h_u32 = unsafe { vminq_u32(vcvtaq_u32_f32(hue_half), vdupq_n_u32(179)) };
          let s_u32 = unsafe { vminq_u32(vcvtaq_u32_f32(sat), vdupq_n_u32(255)) };
          let v_u32 = unsafe { vminq_u32(vcvtaq_u32_f32(val), vdupq_n_u32(255)) };
          (h_u32, s_u32, v_u32)
        }};
      }

      let g0 = process_group!(b_lo16, g_lo16, r_lo16, vmovl_u16_low);
      let g1 = process_group!(b_lo16, g_lo16, r_lo16, vmovl_u16_high);
      let g2 = process_group!(b_hi16, g_hi16, r_hi16, vmovl_u16_low);
      let g3 = process_group!(b_hi16, g_hi16, r_hi16, vmovl_u16_high);

      let h_bufs: [uint32x4_t; 4] = [g0.0, g1.0, g2.0, g3.0];
      let s_bufs: [uint32x4_t; 4] = [g0.1, g1.1, g2.1, g3.1];
      let v_bufs: [uint32x4_t; 4] = [g0.2, g1.2, g2.2, g3.2];

      let h_u8x16 = unsafe { pack_u32x4_quad_to_u8x16(&h_bufs) };
      let s_u8x16 = unsafe { pack_u32x4_quad_to_u8x16(&s_bufs) };
      let v_u8x16 = unsafe { pack_u32x4_quad_to_u8x16(&v_bufs) };
      unsafe {
        vst1q_u8(h_out.as_mut_ptr().add(dst_off + x), h_u8x16);
        vst1q_u8(s_out.as_mut_ptr().add(dst_off + x), s_u8x16);
        vst1q_u8(v_out.as_mut_ptr().add(dst_off + x), v_u8x16);
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

/// Widen the low four lanes of a `uint16x8_t` to `uint32x4_t`.
#[target_feature(enable = "neon")]
#[allow(unused_unsafe)]
#[inline]
unsafe fn vmovl_u16_low(v: uint16x8_t) -> uint32x4_t {
  unsafe { vmovl_u16(vget_low_u16(v)) }
}

/// Widen the high four lanes of a `uint16x8_t` to `uint32x4_t`.
#[target_feature(enable = "neon")]
#[allow(unused_unsafe)]
#[inline]
unsafe fn vmovl_u16_high(v: uint16x8_t) -> uint32x4_t {
  unsafe { vmovl_high_u16(v) }
}

/// Four `u32x4` → one `u8x16`, via saturating narrow. Lane order is
/// preserved: `[q[0][0..4], q[1][0..4], q[2][0..4], q[3][0..4]]`.
#[target_feature(enable = "neon")]
#[allow(unused_unsafe)]
#[inline]
unsafe fn pack_u32x4_quad_to_u8x16(quads: &[uint32x4_t; 4]) -> uint8x16_t {
  let u16_0 = unsafe { vqmovn_u32(quads[0]) };
  let u16_1 = unsafe { vqmovn_u32(quads[1]) };
  let u16_2 = unsafe { vqmovn_u32(quads[2]) };
  let u16_3 = unsafe { vqmovn_u32(quads[3]) };
  let u16_lo = unsafe { vcombine_u16(u16_0, u16_1) };
  let u16_hi = unsafe { vcombine_u16(u16_2, u16_3) };
  let u8_lo = unsafe { vqmovn_u16(u16_lo) };
  let u8_hi = unsafe { vqmovn_u16(u16_hi) };
  unsafe { vcombine_u8(u8_lo, u8_hi) }
}

/// Branch-free 4-lane BGR→HSV core. Returns `(hue ∈ [0, 360),
/// sat ∈ [0, 255], val ∈ [0, 255])` as `f32x4`.
#[target_feature(enable = "neon")]
#[allow(unused_unsafe)]
#[inline]
unsafe fn bgr_to_hsv_f32x4(
  b: float32x4_t,
  g: float32x4_t,
  r: float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t) {
  let zero = unsafe { vdupq_n_f32(0.0) };
  let one = unsafe { vdupq_n_f32(1.0) };

  let v = unsafe { vmaxq_f32(vmaxq_f32(b, g), r) };
  let min = unsafe { vminq_f32(vminq_f32(b, g), r) };
  let delta = unsafe { vsubq_f32(v, min) };

  let delta_zero = unsafe { vceqq_f32(delta, zero) };
  let v_zero = unsafe { vceqq_f32(v, zero) };
  let delta_safe = unsafe { vbslq_f32(delta_zero, one, delta) };

  let sixty = unsafe { vdupq_n_f32(60.0) };
  let c120 = unsafe { vdupq_n_f32(120.0) };
  let c240 = unsafe { vdupq_n_f32(240.0) };
  let c360 = unsafe { vdupq_n_f32(360.0) };
  let c255 = unsafe { vdupq_n_f32(255.0) };

  let h_r = unsafe { vdivq_f32(vmulq_f32(sixty, vsubq_f32(g, b)), delta_safe) };
  let h_g = unsafe {
    vaddq_f32(
      vdivq_f32(vmulq_f32(sixty, vsubq_f32(b, r)), delta_safe),
      c120,
    )
  };
  let h_b = unsafe {
    vaddq_f32(
      vdivq_f32(vmulq_f32(sixty, vsubq_f32(r, g)), delta_safe),
      c240,
    )
  };

  let is_r = unsafe { vceqq_f32(v, r) };
  let is_g = unsafe { vceqq_f32(v, g) };
  let not_r_and_g = unsafe { vandq_u32(vmvnq_u32(is_r), is_g) };
  let hue_rg = unsafe { vbslq_f32(is_r, h_r, h_b) };
  let hue = unsafe { vbslq_f32(not_r_and_g, h_g, hue_rg) };
  let neg = unsafe { vcltq_f32(hue, zero) };
  let hue = unsafe { vbslq_f32(neg, vaddq_f32(hue, c360), hue) };
  let hue = unsafe { vbslq_f32(delta_zero, zero, hue) };

  let v_safe = unsafe { vbslq_f32(v_zero, one, v) };
  let sat = unsafe { vdivq_f32(vmulq_f32(c255, delta), v_safe) };
  let sat = unsafe { vbslq_f32(v_zero, zero, sat) };

  (hue, sat, v)
}

/// NEON `mean_abs_diff`: `Σ|a[i] - b[i]| / n`.
///
/// Uses `vabdq_u8` (absolute-difference, 16 bytes) → `vpaddlq_u8` (pairwise
/// add-long u8→u16) → `vpaddlq_u16` (u16→u32) → `vpaddlq_u32` (u32→u64),
/// accumulating into a `u64x2`. Tail handled scalar.
///
/// # Safety
///
/// Caller must ensure NEON is available (always true on aarch64).
#[target_feature(enable = "neon")]
#[allow(unused_unsafe)]
pub(super) unsafe fn mean_abs_diff(a: &[u8], b: &[u8], n: usize) -> f64 {
  const LANES: usize = 16;
  let whole = n / LANES * LANES;
  let mut acc = unsafe { vdupq_n_u64(0) }; // u64x2 accumulator

  let mut i = 0;
  while i < whole {
    let va = unsafe { vld1q_u8(a.as_ptr().add(i)) };
    let vb = unsafe { vld1q_u8(b.as_ptr().add(i)) };
    // |a - b| as u8x16.
    let diff = unsafe { vabdq_u8(va, vb) };
    // Widen + reduce: u8x16 → u16x8 → u32x4 → u64x2, each step pairwise-sums.
    let s16 = unsafe { vpaddlq_u8(diff) };
    let s32 = unsafe { vpaddlq_u16(s16) };
    let s64 = unsafe { vpaddlq_u32(s32) };
    acc = unsafe { vaddq_u64(acc, s64) };
    i += LANES;
  }

  // Horizontal reduce u64x2 → u64.
  let mut sum: u64 = unsafe { vgetq_lane_u64::<0>(acc) + vgetq_lane_u64::<1>(acc) };

  // Scalar tail.
  while i < n {
    let da = a[i] as i32 - b[i] as i32;
    sum += da.unsigned_abs() as u64;
    i += 1;
  }

  sum as f64 / n as f64
}

/// NEON Sobel 3×3. Computes Gx, Gy, magnitude in i16x8 (8 pixels/iter)
/// via shifted row loads. Direction quantization is scalar from extracted lanes.
///
/// # Safety
///
/// Caller must ensure NEON is available (always true on aarch64).
#[target_feature(enable = "neon")]
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

    // SIMD body: 8 pixels per iteration.
    while x + LANES < w {
      // 9 shifted loads, widen u8x8 → i16x8.
      macro_rules! ld {
        ($row:expr, $o:expr) => {{ unsafe { vreinterpretq_s16_u16(vmovl_u8(vld1_u8($row.as_ptr().add($o)))) } }};
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
        let pos = vaddq_s16(vaddq_s16(pr, vshlq_n_s16::<1>(cr)), nr);
        let neg = vaddq_s16(vaddq_s16(pl, vshlq_n_s16::<1>(cl)), nl);
        vsubq_s16(pos, neg)
      };

      // Gy = (nl + 2*nm + nr) - (pl + 2*pm + pr)
      let gy = unsafe {
        let pos = vaddq_s16(vaddq_s16(nl, vshlq_n_s16::<1>(nm)), nr);
        let neg = vaddq_s16(vaddq_s16(pl, vshlq_n_s16::<1>(pm)), pr);
        vsubq_s16(pos, neg)
      };

      // mag = |gx| + |gy| as i16, then widen to i32 and store.
      let mag_i16 = unsafe { vaddq_s16(vabsq_s16(gx), vabsq_s16(gy)) };
      unsafe {
        vst1q_s32(
          mag.as_mut_ptr().add(off + x),
          vmovl_s16(vget_low_s16(mag_i16)),
        );
        vst1q_s32(mag.as_mut_ptr().add(off + x + 4), vmovl_high_s16(mag_i16));
      }

      // Direction: extract to scalar for the branchy quantization.
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
