//! Criterion benchmark for the content detector across its three hot
//! configurations:
//!
//! 1. `process_luma` with luma-only weights, no edges — the cheapest path.
//! 2. `process_bgr` with default weights, no edges — includes BGR→HSV
//!    conversion.
//! 3. `process_bgr` with default weights + `delta_edges = 1.0` — adds the
//!    full Canny + dilate pipeline.
//!
//! These three numbers pinpoint where the per-frame time actually goes and
//! tell us whether SIMD / algorithmic wins are worth chasing on a given
//! config.
//!
//! Run with `cargo bench --bench content`.

use core::num::NonZeroU32;
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

use scenesdetect::content::{Components, DEFAULT_WEIGHTS, Detector, LUMA_ONLY_WEIGHTS, Options};
use scenesdetect::frame::{LumaFrame, RgbFrame, Timebase, Timestamp};

fn make_buf(n: usize) -> Vec<u8> {
  let mut state: u32 = 0x9E3779B9;
  let mut buf = Vec::with_capacity(n);
  for _ in 0..n {
    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    buf.push((state >> 24) as u8);
  }
  buf
}

fn bench_luma_only(c: &mut Criterion) {
  let tb = Timebase::new(1, NonZeroU32::new(1000).unwrap());
  let mut group = c.benchmark_group("content::Detector::process_luma (luma-only weights)");
  for &(label, w, h) in &[
    ("720p", 1280u32, 720u32),
    ("1080p", 1920u32, 1080u32),
    ("4K", 3840u32, 2160u32),
  ] {
    let buf = make_buf((w * h) as usize);
    group.throughput(criterion::Throughput::Bytes(buf.len() as u64));
    group.bench_function(label, |b| {
      let opts = Options::default().with_weights(LUMA_ONLY_WEIGHTS);
      let mut det = Detector::new(opts);
      let mut pts: i64 = 0;
      b.iter(|| {
        let frame = LumaFrame::new(&buf, w, h, w, Timestamp::new(pts, tb));
        pts += 33;
        black_box(det.process_luma(frame));
      });
    });
  }
  group.finish();
}

fn bench_bgr_no_edges(c: &mut Criterion) {
  let tb = Timebase::new(1, NonZeroU32::new(1000).unwrap());
  let mut group = c.benchmark_group("content::Detector::process_bgr (default weights, no edges)");
  for &(label, w, h) in &[
    ("720p", 1280u32, 720u32),
    ("1080p", 1920u32, 1080u32),
    ("4K", 3840u32, 2160u32),
  ] {
    let buf = make_buf((w * h * 3) as usize);
    group.throughput(criterion::Throughput::Bytes(buf.len() as u64));
    group.bench_function(label, |b| {
      let opts = Options::default().with_weights(DEFAULT_WEIGHTS);
      let mut det = Detector::new(opts);
      let mut pts: i64 = 0;
      b.iter(|| {
        let frame = RgbFrame::new(&buf, w, h, w * 3, Timestamp::new(pts, tb));
        pts += 33;
        black_box(det.process_bgr(frame));
      });
    });
  }
  group.finish();
}

fn bench_bgr_with_edges(c: &mut Criterion) {
  let tb = Timebase::new(1, NonZeroU32::new(1000).unwrap());
  let mut group = c.benchmark_group("content::Detector::process_bgr (with edges)");
  for &(label, w, h) in &[
    ("720p", 1280u32, 720u32),
    ("1080p", 1920u32, 1080u32),
    ("4K", 3840u32, 2160u32),
  ] {
    let buf = make_buf((w * h * 3) as usize);
    group.throughput(criterion::Throughput::Bytes(buf.len() as u64));
    group.bench_function(label, |b| {
      // Equal weights for H/S/V/edges to exercise the full edge pipeline.
      let weights = Components::new(1.0, 1.0, 1.0, 1.0);
      let opts = Options::default().with_weights(weights);
      let mut det = Detector::new(opts);
      let mut pts: i64 = 0;
      b.iter(|| {
        let frame = RgbFrame::new(&buf, w, h, w * 3, Timestamp::new(pts, tb));
        pts += 33;
        black_box(det.process_bgr(frame));
      });
    });
  }
  group.finish();
}

criterion_group!(
  benches,
  bench_luma_only,
  bench_bgr_no_edges,
  bench_bgr_with_edges,
);
criterion_main!(benches);
