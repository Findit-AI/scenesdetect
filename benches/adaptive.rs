//! Criterion benchmark for the adaptive (rolling-average) detector.
//!
//! The adaptive detector is a thin layer over the content detector — each
//! incoming frame goes through the full content scoring path, then the
//! adaptive layer adds a ring-buffer push + mean-over-window computation.
//! The interesting question these numbers answer is "how much overhead does
//! the adaptive layer add on top of the content scorer?"
//!
//! Run with `cargo bench --bench adaptive`.

use core::num::NonZeroU32;
use core::time::Duration;
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

use scenesdetect::adaptive::{Detector, Options};
use scenesdetect::content::{DEFAULT_WEIGHTS, LUMA_ONLY_WEIGHTS};
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
  let mut group = c.benchmark_group("adaptive::Detector::process_luma (luma-only weights)");
  for &(label, w, h) in &[
    ("720p", 1280u32, 720u32),
    ("1080p", 1920u32, 1080u32),
    ("4K", 3840u32, 2160u32),
  ] {
    let buf = make_buf((w * h) as usize);
    group.throughput(criterion::Throughput::Bytes(buf.len() as u64));
    group.bench_function(label, |b| {
      let opts = Options::default()
        .with_weights(LUMA_ONLY_WEIGHTS)
        .with_min_duration(Duration::from_millis(0));
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
  let mut group = c.benchmark_group("adaptive::Detector::process_bgr (default weights, no edges)");
  for &(label, w, h) in &[
    ("720p", 1280u32, 720u32),
    ("1080p", 1920u32, 1080u32),
    ("4K", 3840u32, 2160u32),
  ] {
    let buf = make_buf((w * h * 3) as usize);
    group.throughput(criterion::Throughput::Bytes(buf.len() as u64));
    group.bench_function(label, |b| {
      let opts = Options::default()
        .with_weights(DEFAULT_WEIGHTS)
        .with_min_duration(Duration::from_millis(0));
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

fn bench_window_sizes(c: &mut Criterion) {
  // Isolates the cost of the adaptive layer itself: same luma-only scoring,
  // varying window_width so the ring-buffer sweep grows.
  let tb = Timebase::new(1, NonZeroU32::new(1000).unwrap());
  let mut group = c.benchmark_group("adaptive::Detector::process_luma (1080p, varying window)");
  let (w, h) = (1920u32, 1080u32);
  let buf = make_buf((w * h) as usize);
  group.throughput(criterion::Throughput::Bytes(buf.len() as u64));
  for &window in &[1u32, 2, 4, 8, 16] {
    group.bench_function(format!("window_width={window}"), |b| {
      let opts = Options::default()
        .with_weights(LUMA_ONLY_WEIGHTS)
        .with_window_width(window)
        .with_min_duration(Duration::from_millis(0));
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

criterion_group!(
  benches,
  bench_luma_only,
  bench_bgr_no_edges,
  bench_window_sizes
);
criterion_main!(benches);
