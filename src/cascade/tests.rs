use core::num::{NonZeroU32, NonZeroUsize};

use std::vec;

use super::*;
use crate::frame::Timebase;

fn tb() -> Timebase {
  Timebase::new(1, NonZeroU32::new(1_000).expect("non-zero"))
}

fn ts(ms: i64) -> Timestamp {
  Timestamp::new(ms, tb())
}

/// One synthetic frame: solid luma / hue / sat / val planes plus a
/// solid RGB buffer, 64x48.
struct Fixture {
  rgb: Vec<u8>,
  luma: Vec<u8>,
  hue: Vec<u8>,
  sat: Vec<u8>,
  val: Vec<u8>,
}

const W: u32 = 64;
const H: u32 = 48;
const N: usize = (W * H) as usize;

impl Fixture {
  fn solid(luma: u8, hue: u8, sat: u8) -> Self {
    Self {
      rgb: vec![luma; N * 3],
      luma: vec![luma; N],
      hue: vec![hue; N],
      sat: vec![sat; N],
      val: vec![luma; N],
    }
  }

  /// A textured variant: vertical stripes in luma/RGB so sharpness
  /// and variance metrics see real edges.
  fn striped(dark: u8, light: u8, hue: u8, sat: u8) -> Self {
    let mut luma = vec![0u8; N];
    let mut rgb = vec![0u8; N * 3];
    for row in 0..H as usize {
      for col in 0..W as usize {
        let v = if (col / 4) % 2 == 0 { dark } else { light };
        luma[row * W as usize + col] = v;
        let p = (row * W as usize + col) * 3;
        rgb[p] = v;
        rgb[p + 1] = v;
        rgb[p + 2] = v;
      }
    }
    Self {
      val: luma.clone(),
      luma,
      rgb,
      hue: vec![hue; N],
      sat: vec![sat; N],
    }
  }

  fn frames(&self, at: Timestamp) -> Frames<'_> {
    Frames::try_new(
      RgbFrame::new(&self.rgb, W, H, W * 3, at),
      LumaFrame::new(&self.luma, W, H, W, at),
      HsvFrame::new(&self.hue, &self.sat, &self.val, W, H, W, at),
    )
    .expect("consistent fixture views")
  }
}

fn detector(options: Options) -> Detector {
  Detector::try_new(options).expect("valid options")
}

/// Push a run of identical frames, collecting outputs.
fn push_run(det: &mut Detector, fix: &Fixture, from_ms: i64, count: i64, step: i64) -> Vec<Event> {
  let mut out = Vec::new();
  for i in 0..count {
    if let Some(o) = det
      .push(fix.frames(ts(from_ms + i * step)))
      .expect("ordered stream")
    {
      out.push(o);
    }
  }
  out
}

fn scenes(outputs: &[Event]) -> Vec<SceneEvent> {
  outputs
    .iter()
    .filter_map(|o| match o {
      Event::Scene(s) => Some(s.clone()),
      Event::Keyframe(_) => None,
    })
    .collect()
}

fn keyframes(outputs: &[Event]) -> Vec<KeyframeEvent> {
  outputs
    .iter()
    .filter_map(|o| match o {
      Event::Keyframe(k) => Some(k.clone()),
      Event::Scene(_) => None,
    })
    .collect()
}

/// Options tuned for the synthetic fixtures: no fade lane (solid
/// mid-gray fixtures sit above any fade floor anyway), 100 ms
/// intervals, no adaptive lane (its window lag needs more context than
/// these short fixtures provide).
fn fixture_options() -> Options {
  Options::new()
    .with_detectors(Detectors::default() - Detectors::THRESHOLD - Detectors::ADAPTIVE)
    .with_select(select::Options::new().with_target_interval(Duration::from_millis(100)))
}

#[test]
fn frames_try_new_rejects_mismatches() {
  let a = Fixture::solid(100, 10, 10);
  let rgb = RgbFrame::new(&a.rgb, W, H, W * 3, ts(0));
  let luma = LumaFrame::new(&a.luma, W, H, W, ts(0));
  let hsv_bad_ts = HsvFrame::new(&a.hue, &a.sat, &a.val, W, H, W, ts(1));
  assert_eq!(
    Frames::try_new(rgb, luma, hsv_bad_ts).expect_err("ts mismatch"),
    FramesError::TimestampMismatch
  );
  let half = (W / 2 * H) as usize;
  let small = LumaFrame::new(&a.luma[..half], W / 2, H, W / 2, ts(0));
  let hsv = HsvFrame::new(&a.hue, &a.sat, &a.val, W, H, W, ts(0));
  assert_eq!(
    Frames::try_new(rgb, small, hsv).expect_err("dim mismatch"),
    FramesError::DimensionMismatch
  );

  let empty: [u8; 0] = [];
  let zero = Frames::try_new(
    RgbFrame::new(&empty, 0, H, 0, ts(0)),
    LumaFrame::new(&empty, 0, H, 0, ts(0)),
    HsvFrame::new(&empty, &empty, &empty, 0, H, 0, ts(0)),
  );
  assert_eq!(
    zero.expect_err("zero width"),
    FramesError::ZeroDimensions,
    "zero-sized views have nothing to analyze"
  );
}

#[test]
fn options_validation() {
  let Err(e) = Detector::try_new(Options::new().with_detectors(Detectors::empty())) else {
    panic!("no enabled lane must be rejected");
  };
  assert_eq!(e, OptionsError::NoCutLane);

  // The fade lane alone is a valid composition (fade-only use), and
  // so is a fully default set.
  assert!(Detector::try_new(Options::new().with_detectors(Detectors::THRESHOLD)).is_ok());
  assert!(Detector::try_new(Options::new()).is_ok());

  // Every enabled fallible lane is validated up front: a panic-on-new
  // configuration must surface as an OptionsError instead.
  let zero_weights = Options::new().with_content(
    content::Options::new().with_weights(content::Components::new(0.0, 0.0, 0.0, 0.0)),
  );
  let Err(e) = Detector::try_new(zero_weights) else {
    panic!("zero content weights must be rejected");
  };
  assert!(e.is_content(), "got {e:?}");

  // An absurd adaptive window may not push the lag horizon past the
  // cap.
  let wide = Options::new().with_adaptive(adaptive::Options::new().with_window_width(100_000));
  let Err(e) = Detector::try_new(wide) else {
    panic!("absurd adaptive window must be rejected");
  };
  assert_eq!(e, OptionsError::WindowTooWide(100_000));

  // The content / adaptive lanes are cut lanes like any other.
  let lanes = Detectors::CONTENT | Detectors::ADAPTIVE;
  assert!(Detector::try_new(Options::new().with_detectors(lanes)).is_ok());
}

#[test]
fn consecutive_cuts_are_found() {
  // Two transitions far enough apart for every lane's min_duration:
  // each must surface, with per-candidate admissibility filtering so a
  // stale lagged report can never mask a valid later one at merge.
  let options = Options::new()
    .with_detectors(Detectors::all())
    .with_select(select::Options::new().with_target_interval(Duration::from_millis(100)));
  let mut det = detector(options);
  let red = Fixture::solid(60, 10, 200);
  let blue = Fixture::solid(180, 120, 200);

  let mut outputs = push_run(&mut det, &red, 0, 30, 40);
  outputs.extend(push_run(&mut det, &blue, 30 * 40, 30, 40));
  outputs.extend(push_run(&mut det, &red, 60 * 40, 30, 40));
  outputs.extend(det.finalize());

  let scenes = scenes(&outputs);
  assert_eq!(scenes.len(), 3, "both cuts plus the trailing shot");
  assert_eq!(scenes[0].range().end().pts(), 30 * 40);
  assert_eq!(scenes[1].range().end().pts(), 60 * 40);
}

#[test]
fn finalize_surfaces_the_terminal_fade_out_cut() {
  // A stream ending mid fade-out with add_final_scene owes one last
  // threshold cut; finalize must collect it from the lane's finish
  // hook and close at that boundary before the trailing scene.
  let options = Options::new()
    .with_detectors(Detectors::THRESHOLD)
    .with_threshold(threshold::Options::new().with_add_final_scene(true))
    .with_select(select::Options::new().with_target_interval(Duration::from_millis(100)));
  let mut det = detector(options);
  let bright = Fixture::solid(200, 10, 50);
  let black = Fixture::solid(3, 10, 0);

  let mut outputs = push_run(&mut det, &bright, 0, 15, 40);
  outputs.extend(push_run(&mut det, &black, 15 * 40, 10, 40));
  outputs.extend(det.finalize());

  let scenes = scenes(&outputs);
  assert_eq!(
    scenes.len(),
    2,
    "the fade-out closes one scene, the stream end the other: {scenes:?}"
  );
  assert!(
    scenes[0].provenance_ref().is_threshold(),
    "the first boundary is the terminal fade cut: {:?}",
    scenes[0].provenance_ref()
  );
  assert!(scenes[1].provenance_ref().is_finalized());
}

#[test]
fn adaptive_first_boundary_honors_initial_cut_false() {
  // initial_cut(false) anchors adaptive's first-cut spacing at the
  // stream start; the machine must honor that even though the lane
  // runs with its internal min_duration zeroed.
  let early_cut = |initial_cut: bool| {
    let options = Options::new()
      .with_detectors(Detectors::ADAPTIVE)
      .with_adaptive(adaptive::Options::new().with_initial_cut(initial_cut))
      .with_select(select::Options::new().with_target_interval(Duration::from_millis(100)));
    let mut det = detector(options);
    let red = Fixture::solid(60, 10, 200);
    let blue = Fixture::solid(180, 120, 200);

    // The flip lands 400 ms in — before adaptive's 1 s min_duration
    // measured from the stream start.
    let mut outputs = push_run(&mut det, &red, 0, 10, 40);
    outputs.extend(push_run(&mut det, &blue, 10 * 40, 20, 40));
    outputs.extend(det.finalize());
    scenes(&outputs).len()
  };

  assert_eq!(
    early_cut(false),
    1,
    "initial_cut(false): the early first cut is suppressed"
  );
  assert_eq!(
    early_cut(true),
    2,
    "initial_cut(true): the early first cut is allowed"
  );
}

#[test]
fn content_first_boundary_honors_initial_cut_false() {
  // The same warm-up contract as adaptive, for the content lane:
  // initial_cut(false) anchors content's first-cut spacing at the
  // stream start, and the machine must honor it even though the lane
  // runs with its internal min_duration zeroed.
  let early_cut = |initial_cut: bool| {
    let options = Options::new()
      .with_detectors(Detectors::CONTENT)
      .with_content(content::Options::new().with_initial_cut(initial_cut))
      .with_select(select::Options::new().with_target_interval(Duration::from_millis(100)));
    let mut det = detector(options);
    let red = Fixture::solid(60, 10, 200);
    let blue = Fixture::solid(180, 120, 200);

    let mut outputs = push_run(&mut det, &red, 0, 10, 40);
    outputs.extend(push_run(&mut det, &blue, 10 * 40, 20, 40));
    outputs.extend(det.finalize());
    scenes(&outputs).len()
  };

  assert_eq!(
    early_cut(false),
    1,
    "initial_cut(false): the early first content cut is suppressed"
  );
  assert_eq!(
    early_cut(true),
    2,
    "initial_cut(true): the early first content cut is allowed"
  );
}

#[test]
fn dimension_reset_re_anchors_first_cut_gating() {
  // After a resolution switch the lanes' state is cleared; a lane
  // configured with initial_cut(false) must warm up against the NEW
  // visual stream, not the pre-reset timeline. A transition 400 ms
  // after the reset sits 2.4 s into the stream — far past min_duration
  // from the old anchor, but inside the warm-up from the reset.
  fn shrink(f: &Fixture, at: Timestamp) -> Frames<'_> {
    let n = (W / 2 * H) as usize;
    Frames::try_new(
      RgbFrame::new(&f.rgb[..n * 3], W / 2, H, W / 2 * 3, at),
      LumaFrame::new(&f.luma[..n], W / 2, H, W / 2, at),
      HsvFrame::new(&f.hue[..n], &f.sat[..n], &f.val[..n], W / 2, H, W / 2, at),
    )
    .expect("consistent half views")
  }
  let run = |initial_cut: bool| {
    let options = Options::new()
      .with_detectors(Detectors::ADAPTIVE)
      .with_adaptive(adaptive::Options::new().with_initial_cut(initial_cut))
      .with_select(select::Options::new().with_target_interval(Duration::from_millis(100)));
    let mut det = detector(options);
    let red = Fixture::solid(60, 10, 200);
    let blue = Fixture::solid(180, 120, 200);

    let mut outputs = push_run(&mut det, &red, 0, 50, 40);
    for i in 0..10 {
      if let Some(o) = det
        .push(shrink(&red, ts(2000 + i * 40)))
        .expect("ordered stream")
      {
        outputs.push(o);
      }
    }
    for i in 0..20 {
      if let Some(o) = det
        .push(shrink(&blue, ts(2400 + i * 40)))
        .expect("ordered stream")
      {
        outputs.push(o);
      }
    }
    outputs.extend(det.finalize());
    scenes(&outputs).len()
  };

  assert_eq!(
    run(false),
    1,
    "initial_cut(false): the early post-reset cut is suppressed"
  );
  assert_eq!(
    run(true),
    2,
    "initial_cut(true): the post-reset cut is allowed"
  );
}

#[test]
fn frames_reject_oversized_views() {
  // Every enabled lane scans the frame each push; frame area is
  // bounded up front so per-push compute has a ceiling.
  let w: u32 = 2048;
  let h: u32 = 1024;
  let n = (w * h) as usize;
  let rgb = vec![0u8; n * 3];
  let plane = vec![0u8; n];
  let frames = Frames::try_new(
    RgbFrame::new(&rgb, w, h, w * 3, ts(0)),
    LumaFrame::new(&plane, w, h, w, ts(0)),
    HsvFrame::new(&plane, &plane, &plane, w, h, w, ts(0)),
  );
  assert_eq!(
    frames.expect_err("past the pixel cap"),
    FramesError::FrameTooLarge {
      pixels: u64::from(w) * u64::from(h)
    }
  );
}

#[test]
fn frames_reject_zero_numerator_timebases() {
  // A zero-numerator timebase names the same instant for every pts;
  // boundary ranges would need a rescale that panics downstream, so
  // the bundle rejects it up front.
  let fix = Fixture::solid(100, 10, 50);
  let degenerate = Timestamp::new(0, Timebase::new(0, NonZeroU32::new(1_000).expect("nz")));
  let frames = Frames::try_new(
    RgbFrame::new(&fix.rgb, W, H, W * 3, degenerate),
    LumaFrame::new(&fix.luma, W, H, W, degenerate),
    HsvFrame::new(&fix.hue, &fix.sat, &fix.val, W, H, W, degenerate),
  );
  assert_eq!(
    frames.expect_err("degenerate timebase"),
    FramesError::ZeroTimebase
  );

  // A degenerate timebase hiding in a SIDE view must be caught too:
  // semantic equality would otherwise accept a zero-numerator luma or
  // HSV timestamp that compares equal at instant zero.
  let healthy = Timestamp::new(0, Timebase::new(1, NonZeroU32::new(1_000).expect("nz")));
  let mixed = Frames::try_new(
    RgbFrame::new(&fix.rgb, W, H, W * 3, healthy),
    LumaFrame::new(&fix.luma, W, H, W, degenerate),
    HsvFrame::new(&fix.hue, &fix.sat, &fix.val, W, H, W, healthy),
  );
  assert_eq!(
    mixed.expect_err("degenerate side-view timebase"),
    FramesError::ZeroTimebase
  );
  let mixed_hsv = Frames::try_new(
    RgbFrame::new(&fix.rgb, W, H, W * 3, healthy),
    LumaFrame::new(&fix.luma, W, H, W, healthy),
    HsvFrame::new(&fix.hue, &fix.sat, &fix.val, W, H, W, degenerate),
  );
  assert_eq!(
    mixed_hsv.expect_err("degenerate hsv timebase"),
    FramesError::ZeroTimebase
  );
}

#[test]
fn pathological_boundary_rate_keeps_the_backlog_bounded() {
  // Zero adaptive threshold, min-content and min-duration are
  // type-valid and confirm a boundary on virtually every push after
  // warm-up; each boundary enqueues up to a keyframe plus a scene
  // while push drains one output. The backlog must stay bounded by
  // shedding oldest keyframes — and scenes must all survive.
  let options = Options::new()
    .with_detectors(Detectors::ADAPTIVE)
    .with_adaptive(
      adaptive::Options::new()
        .with_adaptive_threshold(0.0)
        .with_min_content_val(0.0)
        .with_min_duration(Duration::ZERO),
    )
    .with_select(select::Options::new().with_target_interval(Duration::from_millis(100)));
  let mut det = detector(options);
  let a = Fixture::solid(60, 10, 200);
  let b = Fixture::solid(180, 120, 200);

  let mut delivered = 0usize;
  let mut backlog_max = 0usize;
  for i in 0..600_i64 {
    let fix = if i % 2 == 0 { &a } else { &b };
    if det
      .push(fix.frames(ts(i * 40)))
      .expect("ordered stream")
      .is_some()
    {
      delivered += 1;
    }
    backlog_max = backlog_max.max(det.pending.len());
  }
  let tail = det.finalize();

  assert!(
    backlog_max <= 257,
    "the backlog stays at the cap, got {backlog_max}"
  );
  assert!(delivered > 0, "outputs keep flowing");
  let scene_count = tail.iter().filter(|o| o.is_scene()).count();
  assert!(
    scene_count > 0 || delivered > 0,
    "scenes survive the shedding"
  );
}

#[test]
fn mixed_timebases_keep_scene_ranges_partitioned() {
  // Pushes may carry different (valid) timebases; every emitted range
  // must come out in the stream's canonical timebase with consecutive
  // ranges exactly adjacent — a per-boundary timebase would truncate
  // the saved start and overlap the previous scene.
  let options = Options::new()
    .with_detectors(Detectors::THRESHOLD)
    .with_max_scene_duration(Some(Duration::from_millis(200)))
    .with_select(select::Options::new().with_target_interval(Duration::from_millis(100)));
  let mut det = detector(options);
  let fix = Fixture::striped(40, 220, 20, 180);

  let ms = tb();
  let coarse = Timebase::new(1, NonZeroU32::new(1).expect("nz"));
  let mut outputs = Vec::new();
  // 800 ms of millisecond-timebase frames…
  for i in 0..20_i64 {
    if let Some(o) = det
      .push(fix.frames(Timestamp::new(i * 40, ms)))
      .expect("ordered stream")
    {
      outputs.push(o);
    }
  }
  // …then whole-second frames in a 1-second timebase.
  for sec in 1..=4_i64 {
    if let Some(o) = det
      .push(fix.frames(Timestamp::new(sec, coarse)))
      .expect("ordered stream")
    {
      outputs.push(o);
    }
  }
  outputs.extend(det.finalize());

  let scenes = scenes(&outputs);
  assert!(scenes.len() >= 3, "forced splits across both segments");
  for s in &scenes {
    assert_eq!(
      s.range().timebase().den(),
      ms.den(),
      "every range is emitted in the canonical (first) timebase"
    );
  }
  for pair in scenes.windows(2) {
    assert!(
      pair[0]
        .range()
        .end()
        .cmp_semantic(&pair[1].range().start())
        .is_eq(),
      "ranges stay exactly adjacent across the timebase switch: {:?} then {:?}",
      pair[0].range(),
      pair[1].range()
    );
  }
}

#[test]
fn coarse_canonical_timebase_keeps_association_total() {
  // The FIRST frame anchors a one-second canonical timebase; all the
  // content lives inside tick zero. Ceiling-rescaled endpoints keep
  // the trailing range real ([0, 1) rather than a collapsed [0, 0)),
  // so every delivered keyframe still has an owning range.
  let options = Options::new()
    .with_detectors(Detectors::LUMA_HISTOGRAM | Detectors::CONTENT)
    .with_select(select::Options::new().with_target_interval(Duration::from_millis(100)));
  let mut det = detector(options);
  let fix = Fixture::striped(40, 220, 20, 180);

  let coarse = Timebase::new(1, NonZeroU32::new(1).expect("nz"));
  let mut outputs = Vec::new();
  if let Some(o) = det
    .push(fix.frames(Timestamp::new(0, coarse)))
    .expect("ordered stream")
  {
    outputs.push(o);
  }
  for i in 1..20_i64 {
    if let Some(o) = det.push(fix.frames(ts(i * 40))).expect("ordered stream") {
      outputs.push(o);
    }
  }
  outputs.extend(det.finalize());

  let scenes = scenes(&outputs);
  assert_eq!(
    scenes.len(),
    1,
    "the sub-tick stream still emits one ceiling-rounded scene"
  );
  let range = scenes[0].range();
  assert_eq!(range.timebase().den(), coarse.den(), "canonical timebase");
  for k in keyframes(&outputs) {
    assert!(
      k.timestamp().cmp_semantic(&range.start()).is_ge()
        && k.timestamp().cmp_semantic(&range.end()).is_lt(),
      "every keyframe falls inside the emitted range: {:?} vs {range:?}",
      k.timestamp()
    );
  }
  assert!(!keyframes(&outputs).is_empty(), "keyframes were delivered");
}

#[test]
fn dimension_switch_respects_lane_min_duration() {
  // A resolution switch reseeds the continuous lanes' own debounce
  // state; the machine must still enforce each lane's configured
  // min_duration against the timeline, or a reseeded lane could cut
  // again inside its spacing window right after the switch.
  fn shrink(f: &Fixture, at: Timestamp) -> Frames<'_> {
    let n = (W / 2 * H) as usize;
    Frames::try_new(
      RgbFrame::new(&f.rgb[..n * 3], W / 2, H, W / 2 * 3, at),
      LumaFrame::new(&f.luma[..n], W / 2, H, W / 2, at),
      HsvFrame::new(&f.hue[..n], &f.sat[..n], &f.val[..n], W / 2, H, W / 2, at),
    )
    .expect("consistent half views")
  }
  let options = Options::new()
    .with_detectors(Detectors::CONTENT)
    .with_select(select::Options::new().with_target_interval(Duration::from_millis(100)));
  let mut det = detector(options);
  let red = Fixture::solid(60, 10, 200);
  let blue = Fixture::solid(180, 120, 200);

  let mut outputs = push_run(&mut det, &red, 0, 30, 40);
  // Real cut at 1200 ms, then a resolution switch at 1280 ms, then a
  // second transition at 1400 ms — inside content's one-second
  // min_duration from the confirmed cut.
  outputs.extend(push_run(&mut det, &blue, 30 * 40, 2, 40));
  for i in 0..3_i64 {
    if let Some(o) = det
      .push(shrink(&blue, ts(1280 + i * 40)))
      .expect("ordered stream")
    {
      outputs.push(o);
    }
  }
  for i in 0..20_i64 {
    if let Some(o) = det
      .push(shrink(&red, ts(1400 + i * 40)))
      .expect("ordered stream")
    {
      outputs.push(o);
    }
  }
  outputs.extend(det.finalize());

  let scenes = scenes(&outputs);
  assert_eq!(
    scenes.len(),
    2,
    "the post-switch re-cut inside min_duration is suppressed: {:?}",
    scenes.iter().map(|s| s.range()).collect::<Vec<_>>()
  );
  assert_eq!(scenes[0].range().end().pts(), 30 * 40);
}

#[test]
fn rejected_direct_cut_does_not_shadow_a_later_valid_one() {
  // The machine rejects a post-reset cut inside content's
  // min_duration; with the lane's internal debounce zeroed, that
  // rejection must NOT advance any anchor — a later transition that is
  // valid relative to the last CONFIRMED boundary still cuts.
  fn shrink(f: &Fixture, at: Timestamp) -> Frames<'_> {
    let n = (W / 2 * H) as usize;
    Frames::try_new(
      RgbFrame::new(&f.rgb[..n * 3], W / 2, H, W / 2 * 3, at),
      LumaFrame::new(&f.luma[..n], W / 2, H, W / 2, at),
      HsvFrame::new(&f.hue[..n], &f.sat[..n], &f.val[..n], W / 2, H, W / 2, at),
    )
    .expect("consistent half views")
  }
  let options = Options::new()
    .with_detectors(Detectors::CONTENT)
    .with_select(select::Options::new().with_target_interval(Duration::from_millis(100)));
  let mut det = detector(options);
  let red = Fixture::solid(60, 10, 200);
  let blue = Fixture::solid(180, 120, 200);

  // Confirmed cut at 1200 ms, resolution switch at 1280 ms, a REJECTED
  // transition at 1400 ms (inside the 1 s spacing from 1200), then a
  // valid transition at 2280 ms — more than 1 s past the confirmed cut
  // but only 880 ms past the rejected one.
  let mut outputs = push_run(&mut det, &red, 0, 30, 40);
  outputs.extend(push_run(&mut det, &blue, 30 * 40, 2, 40));
  for i in 0..3_i64 {
    if let Some(o) = det
      .push(shrink(&blue, ts(1280 + i * 40)))
      .expect("ordered stream")
    {
      outputs.push(o);
    }
  }
  for i in 0..22_i64 {
    if let Some(o) = det
      .push(shrink(&red, ts(1400 + i * 40)))
      .expect("ordered stream")
    {
      outputs.push(o);
    }
  }
  for i in 0..20_i64 {
    if let Some(o) = det
      .push(shrink(&blue, ts(2280 + i * 40)))
      .expect("ordered stream")
    {
      outputs.push(o);
    }
  }
  outputs.extend(det.finalize());

  let scenes = scenes(&outputs);
  assert_eq!(
    scenes.len(),
    3,
    "the rejected cut neither fires nor shadows the later one: {:?}",
    scenes.iter().map(|s| s.range()).collect::<Vec<_>>()
  );
  assert_eq!(scenes[0].range().end().pts(), 30 * 40);
  assert_eq!(scenes[1].range().end().pts(), 2280);
}

#[test]
fn lagged_boundary_keeps_the_trigger_frame_in_the_new_shot() {
  // Adaptive reports with window lag: the cut lands frames before the
  // push that surfaces it. Keyframes associate to scenes by timestamp,
  // so the frame AT the cut is simply the new scene's first frame and
  // is eligible as one of its keyframes — what matters is that the
  // split lands at the flip frame and the new shot keeps coverage.
  let options = Options::new()
    .with_detectors(Detectors::ADAPTIVE)
    .with_select(select::Options::new().with_target_interval(Duration::from_millis(100)));
  let mut det = detector(options);
  let red = Fixture::solid(60, 10, 200);
  let blue = Fixture::solid(180, 120, 200);

  let mut outputs = push_run(&mut det, &red, 0, 30, 40);
  outputs.extend(push_run(&mut det, &blue, 30 * 40, 30, 40));
  outputs.extend(det.finalize());

  let scenes = scenes(&outputs);
  assert_eq!(scenes.len(), 2, "the lagged adaptive cut still splits");
  assert!(
    scenes[0].provenance_ref().is_adaptive(),
    "attributed to the adaptive lane: {:?}",
    scenes[0].provenance_ref()
  );
  let cut = scenes[0].range().end();
  assert_eq!(cut.pts(), 30 * 40, "the refined cut sits at the flip frame");
  // The new shot keeps coverage even though its first frames were
  // pushed before the boundary surfaced (the flip frame at the cut is
  // now eligible as the new scene's first keyframe).
  let in_second = keyframes(&outputs)
    .iter()
    .filter(|k| k.timestamp().cmp_semantic(&cut).is_ge())
    .count();
  assert!(in_second > 0, "carried + trigger frames cover the new shot");
}

#[test]
fn extractor_flags_prune_expensive_metrics() {
  // Only colorfulness enabled: sharpness is never computed, so the
  // min-sharpness strictness check is skipped and textured frames
  // still produce strict picks whose sharpness reads zero.
  let options = fixture_options().with_extractors(Extractors::COLORFULNESS);
  let mut det = detector(options);
  let fix = Fixture::striped(40, 220, 20, 180);
  let mut outputs = push_run(&mut det, &fix, 0, 12, 33);
  outputs.extend(det.finalize());

  let kfs = keyframes(&outputs);
  assert!(!kfs.is_empty());
  let strict: Vec<_> = kfs
    .iter()
    .filter(|k| k.provenance_ref().is_quality())
    .collect();
  assert!(!strict.is_empty(), "strictness survives without sharpness");
  for k in strict {
    if let Provenance::Quality(m) = k.provenance_ref() {
      assert_eq!(m.sharpness(), 0.0, "disabled extractor stays zero");
    }
  }
}

#[test]
fn every_window_with_frames_yields_keyframes() {
  let mut det = detector(fixture_options());
  // Textured frames pass the variance gates. 30 frames @ 33 ms with a
  // 100 ms window (no adaptive lane, so finalize steps exact windows
  // up to the current frame): each closed 100 ms window yields ONE
  // keyframe. ~990 ms of frames tiles into ~10 windows = 10 keyframes,
  // all in-range and strictly increasing.
  let fix = Fixture::striped(40, 220, 20, 180);
  let mut outputs = push_run(&mut det, &fix, 0, 30, 33);
  outputs.extend(det.finalize());

  let kfs = keyframes(&outputs);
  assert_eq!(
    kfs.len(),
    10,
    "one keyframe per 100 ms window over ~990 ms, got {}",
    kfs.len()
  );
  for pair in kfs.windows(2) {
    assert!(
      pair[0]
        .timestamp()
        .cmp_semantic(&pair[1].timestamp())
        .is_lt(),
      "keyframe timestamps strictly increase"
    );
  }
  let scenes = scenes(&outputs);
  assert_eq!(scenes.len(), 1, "no cut: one finalized scene");
  for k in &kfs {
    assert!(
      k.timestamp()
        .cmp_semantic(&scenes[0].range().start())
        .is_ge()
        && k.timestamp().cmp_semantic(&scenes[0].range().end()).is_lt(),
      "keyframes fall inside the scene range"
    );
  }
}

#[test]
fn gate_failing_intervals_surface_fallback_keyframes() {
  let mut det = detector(fixture_options());
  // Solid frames have zero variance everywhere: the flat AND-gate
  // trips, stage 4 is skipped, and intervals fall back.
  let flat = Fixture::solid(120, 10, 10);
  let mut outputs = push_run(&mut det, &flat, 0, 12, 33);
  outputs.extend(det.finalize());

  let kfs = keyframes(&outputs);
  assert!(!kfs.is_empty(), "flat content still yields coverage");
  for k in &kfs {
    assert!(
      k.provenance_ref().is_fallback(),
      "flat frames can only be fallback picks: {:?}",
      k.provenance_ref()
    );
    if let Provenance::Fallback(m) = k.provenance_ref() {
      assert_eq!(m.sharpness(), 0.0, "expensive metrics were skipped");
      assert!(m.brightness() > 0.0, "cheap metrics were computed");
    }
  }
}

#[test]
fn strict_keyframes_carry_full_metrics() {
  let mut det = detector(fixture_options());
  let fix = Fixture::striped(40, 220, 20, 180);
  let mut outputs = push_run(&mut det, &fix, 0, 12, 33);
  outputs.extend(det.finalize());

  let kfs = keyframes(&outputs);
  assert!(!kfs.is_empty());
  let strict: Vec<_> = kfs
    .iter()
    .filter(|k| k.provenance_ref().is_quality())
    .collect();
  assert!(
    !strict.is_empty(),
    "textured frames pass the gates: {:?}",
    kfs[0].provenance_ref()
  );
  for k in strict {
    if let Provenance::Quality(m) = k.provenance_ref() {
      assert!(m.sharpness() > 0.0, "stage 4 ran for strict picks");
    }
  }
}

#[test]
fn max_scene_duration_force_splits_cutless_content() {
  let options = fixture_options().with_max_scene_duration(Some(Duration::from_millis(200)));
  let mut det = detector(options);
  let fix = Fixture::striped(40, 220, 20, 180);
  let mut outputs = push_run(&mut det, &fix, 0, 30, 33);
  outputs.extend(det.finalize());

  let scenes = scenes(&outputs);
  assert!(
    scenes.len() >= 3,
    "a 200 ms cap over ~1 s of cut-less content forces splits, got {}",
    scenes.len()
  );
  assert!(scenes[0].provenance_ref().is_max_span());
  if let Provenance::MaxSpan(s) = scenes[0].provenance_ref() {
    assert!(s.span() > s.cap());
  }
  // Ranges stay contiguous across forced splits.
  for pair in scenes.windows(2) {
    assert!(
      pair[0]
        .range()
        .end()
        .cmp_semantic(&pair[1].range().start())
        .is_eq(),
      "scenes partition the timeline"
    );
  }
}

#[test]
fn keyframes_precede_their_scene_and_belong_to_its_range() {
  let options = fixture_options().with_max_scene_duration(Some(Duration::from_millis(200)));
  let mut det = detector(options);
  let fix = Fixture::striped(40, 220, 20, 180);
  let mut outputs = push_run(&mut det, &fix, 0, 30, 33);
  outputs.extend(det.finalize());

  // Walk the stream: every keyframe must fall inside the *next*
  // scene range to arrive (ts-based association).
  let mut pending: Vec<Timestamp> = Vec::new();
  for o in &outputs {
    match o {
      Event::Keyframe(k) => pending.push(k.timestamp()),
      Event::Scene(s) => {
        let range = s.range();
        // Keyframes for this scene drain; later ones stay pending.
        pending.retain(|ts| {
          let mine =
            ts.cmp_semantic(&range.start()).is_ge() && ts.cmp_semantic(&range.end()).is_lt();
          !mine
        });
        for stray in &pending {
          assert!(
            stray.cmp_semantic(&range.end()).is_ge(),
            "no keyframe of an already-closed scene may still be pending"
          );
        }
      }
    }
  }
  assert!(pending.is_empty(), "every keyframe's scene arrived");

  // Partition: every keyframe is owned by EXACTLY ONE scene range.
  let scenes = scenes(&outputs);
  for k in keyframes(&outputs) {
    let owners = scenes
      .iter()
      .filter(|sc| {
        k.timestamp().cmp_semantic(&sc.range().start()).is_ge()
          && k.timestamp().cmp_semantic(&sc.range().end()).is_lt()
      })
      .count();
    assert_eq!(
      owners,
      1,
      "keyframe {:?} owned by exactly one scene",
      k.timestamp()
    );
  }
}

#[test]
fn finalize_refreshes_for_the_next_stream() {
  let mut det = detector(fixture_options());
  let fix = Fixture::striped(40, 220, 20, 180);
  let first = {
    let mut outputs = push_run(&mut det, &fix, 0, 6, 33);
    outputs.extend(det.finalize());
    outputs
  };
  assert!(!scenes(&first).is_empty(), "first stream closed");

  // Reused instance: a fresh stream starting at t=0 again.
  let second = {
    let mut outputs = push_run(&mut det, &fix, 0, 6, 33);
    outputs.extend(det.finalize());
    outputs
  };
  let s = scenes(&second);
  assert_eq!(s.len(), 1, "reused detector behaves like a fresh one");
  assert!(s[0].range().start().cmp_semantic(&ts(0)).is_eq());
  assert!(s[0].provenance_ref().is_finalized());
}

#[test]
fn clear_discards_without_emitting() {
  let mut det = detector(fixture_options());
  let fix = Fixture::striped(40, 220, 20, 180);
  let _ = push_run(&mut det, &fix, 0, 8, 33);
  det.clear();
  assert!(det.finalize().is_empty(), "nothing survives clear");
}

#[test]
fn dimension_change_resets_visual_state_only() {
  let mut det = detector(fixture_options());
  let fix = Fixture::striped(40, 220, 20, 180);
  let mut outputs = push_run(&mut det, &fix, 0, 6, 33);

  // Same content at half width: must not register as a cut
  // (cross-size deltas are meaningless), and the shot survives.
  fn small(at: Timestamp, f: &Fixture) -> Frames<'_> {
    let n = (W / 2 * H) as usize;
    Frames::try_new(
      RgbFrame::new(&f.rgb[..n * 3], W / 2, H, W / 2 * 3, at),
      LumaFrame::new(&f.luma[..n], W / 2, H, W / 2, at),
      HsvFrame::new(&f.hue[..n], &f.sat[..n], &f.val[..n], W / 2, H, W / 2, at),
    )
    .expect("consistent half views")
  }
  let half = Fixture::striped(40, 220, 20, 180);
  for i in 6..12 {
    if let Some(o) = det.push(small(ts(i * 33), &half)).expect("ordered stream") {
      outputs.push(o);
    }
  }
  outputs.extend(det.finalize());
  let s = scenes(&outputs);
  assert_eq!(s.len(), 1, "no cut across the dimension change");
  assert!(
    s[0].range().start().cmp_semantic(&ts(0)).is_eq(),
    "the shot survived the dimension change"
  );
}

#[test]
fn flat_stream_emits_no_cuts_but_still_samples_keyframes() {
  // Static content gives every lane a zero delta: no cut fires, so the
  // stream is one finalized shot. Keyframes are still sampled per
  // window (the flat AND-gate trips, so they surface as fallback
  // picks) and must all fall inside that single scene's range.
  let mut det = detector(fixture_options());
  let fix = Fixture::solid(120, 10, 10);
  let mut outputs = push_run(&mut det, &fix, 0, 40, 33);
  outputs.extend(det.finalize());
  let s = scenes(&outputs);
  assert_eq!(s.len(), 1, "static content is one finalized shot");
  assert!(s[0].provenance_ref().is_finalized());
  let kfs = keyframes(&outputs);
  assert!(
    !kfs.is_empty(),
    "a flat stream still emits per-window samples"
  );
  for k in &kfs {
    assert!(
      k.timestamp().cmp_semantic(&s[0].range().start()).is_ge()
        && k.timestamp().cmp_semantic(&s[0].range().end()).is_lt(),
      "every keyframe falls inside the single scene's range"
    );
  }
}

#[test]
fn every_enabled_lane_cuts_directly() {
  // Detectors::all() = every lane. The luma histogram is checked
  // first among the lanes reporting the boundary frame, so the direct
  // cut is attributed to it.
  let options = Options::new()
    .with_detectors(Detectors::all())
    .with_select(select::Options::new().with_target_interval(Duration::from_millis(100)));
  let mut det = detector(options);
  let red = Fixture::solid(60, 10, 200);
  let blue = Fixture::solid(180, 120, 200);

  let mut outputs = push_run(&mut det, &red, 0, 30, 40);
  outputs.extend(push_run(&mut det, &blue, 30 * 40, 30, 40));
  outputs.extend(det.finalize());

  let scenes = scenes(&outputs);
  assert_eq!(scenes.len(), 2, "the full lane set finds the cut");
  if let Provenance::Histogram(score) = scenes[0].provenance_ref() {
    assert!(
      score.plane().is_luma(),
      "earliest equal report wins by lane order"
    );
    assert!(
      score.correlation() < 0.5,
      "the firing correlation sits below the configured threshold"
    );
  } else {
    panic!(
      "expected a direct histogram cut: {:?}",
      scenes[0].provenance_ref()
    );
  }
}

#[test]
fn a_single_lane_is_a_valid_composition() {
  // Every lane cuts directly, so any single lane composes alone.
  let lanes = Detectors::LUMA_HISTOGRAM;
  assert!(Detector::try_new(Options::new().with_detectors(lanes)).is_ok());
}

#[test]
fn chroma_only_cut_is_caught_by_the_saturation_lane() {
  // Luma identical on both sides; only saturation flips. The luma
  // histogram and phash stay silent — the saturation-plane lane is
  // the recall guard and cuts directly at the flip.
  let mut det = detector(fixture_options());
  let gray = Fixture::solid(120, 10, 0);
  let vivid = Fixture::solid(120, 10, 240);

  let mut outputs = push_run(&mut det, &gray, 0, 30, 40);
  outputs.extend(push_run(&mut det, &vivid, 30 * 40, 30, 40));
  outputs.extend(det.finalize());

  let scenes = scenes(&outputs);
  assert_eq!(scenes.len(), 2, "the chroma-only transition is caught");
  assert!(
    matches!(
      scenes[0].provenance_ref(),
      Provenance::Histogram(s) if s.plane().is_saturation()
    ),
    "the saturation lane cut directly: {:?}",
    scenes[0].provenance_ref()
  );
  assert_eq!(
    scenes[0].range().end().pts(),
    30 * 40,
    "cut at the first frame of the new look"
  );
}

#[test]
fn same_push_reports_defer_rather_than_vanish() {
  // A lagged adaptive report and a current-frame histogram report land
  // on ONE push: the earliest closes the shot, and the later one must
  // be deferred and re-gated — its lane consumed the transition
  // evidence, so discarding it would erase the boundary outright.
  let options = Options::new()
    .with_detectors(Detectors::LUMA_HISTOGRAM | Detectors::ADAPTIVE)
    .with_luma_histogram(histogram::Options::new().with_min_duration(Duration::ZERO))
    .with_adaptive(
      adaptive::Options::new()
        .with_window_width(2)
        .with_min_duration(Duration::ZERO),
    )
    .with_select(select::Options::new().with_target_interval(Duration::from_millis(100)));
  let mut det = detector(options);
  let a = Fixture::solid(120, 10, 200);
  let b = Fixture::solid(120, 138, 200); // hue-only flip: luma lane blind
  let c = Fixture::solid(180, 138, 200); // luma flip: adaptive-quiet, lane loud

  // Frames every 40 ms: a for [0,800), b for [800,880), c onward. The
  // adaptive window (2) surfaces the 800 ms hue cut on the SAME push
  // where the luma histogram reports the 880 ms cut.
  let mut outputs = push_run(&mut det, &a, 0, 20, 40);
  outputs.extend(push_run(&mut det, &b, 800, 2, 40));
  outputs.extend(push_run(&mut det, &c, 880, 16, 40));
  outputs.extend(det.finalize());

  let scenes = scenes(&outputs);
  assert_eq!(
    scenes.len(),
    3,
    "both same-push boundaries survive: {:?}",
    scenes.iter().map(|s| s.range()).collect::<Vec<_>>()
  );
  assert_eq!(scenes[0].range().end().pts(), 800);
  assert_eq!(scenes[1].range().end().pts(), 880);
  assert!(scenes[0].provenance_ref().is_adaptive());
  assert!(scenes[1].provenance_ref().is_histogram());
}

#[test]
fn eos_fade_inside_warm_up_is_suppressed() {
  // The end-of-stream fade is freshly observed during finalize: with
  // threshold initial_cut(false), a fade that begins inside the
  // configured warm-up window must be suppressed exactly as it would
  // be mid-stream — finalize's no-warm-up recheck applies only to
  // boundaries that already passed their observation-time gates.
  let options = Options::new()
    .with_detectors(Detectors::THRESHOLD)
    .with_threshold(
      threshold::Options::new()
        .with_initial_cut(false)
        .with_add_final_scene(true),
    )
    .with_select(select::Options::new().with_target_interval(Duration::from_millis(100)));
  let mut det = detector(options);
  let bright = Fixture::solid(160, 10, 10);
  let dim = Fixture::solid(40, 10, 10);
  let black = Fixture::solid(4, 10, 10);

  // The stream ends mid fade-out at 480 ms — well inside the 1 s
  // warm-up of the loose-start contract.
  let mut outputs = push_run(&mut det, &bright, 0, 10, 40);
  outputs.extend(push_run(&mut det, &dim, 400, 1, 40));
  outputs.extend(push_run(&mut det, &black, 440, 2, 40));
  outputs.extend(det.finalize());

  let scenes = scenes(&outputs);
  assert_eq!(
    scenes.len(),
    1,
    "the warm-up suppresses the EOS fade: {:?}",
    scenes.iter().map(|s| s.range()).collect::<Vec<_>>()
  );
  assert!(scenes[0].provenance_ref().is_finalized());
}

#[test]
fn lagged_adaptive_provenance_carries_the_boundary_score() {
  // Adaptive emits window_width frames behind: at the moment the cut
  // is reported the trailing frame is quiet (B→B scores 0). The
  // provenance must carry the TARGET frame's score — the value the
  // adaptive test actually judged — not the trailing frame's.
  let options = Options::new()
    .with_detectors(Detectors::ADAPTIVE)
    .with_adaptive(
      adaptive::Options::new()
        .with_window_width(1)
        .with_min_duration(Duration::ZERO),
    )
    .with_select(select::Options::new().with_target_interval(Duration::from_millis(100)));
  let mut det = detector(options);
  let a = Fixture::solid(160, 10, 10);
  let b = Fixture::solid(60, 10, 10);

  let mut outputs = push_run(&mut det, &a, 0, 30, 40);
  outputs.extend(push_run(&mut det, &b, 1200, 8, 40));
  outputs.extend(det.finalize());

  let scenes = scenes(&outputs);
  assert_eq!(scenes.len(), 2, "the lagged cut closed");
  assert_eq!(scenes[0].range().end().pts(), 1200);
  if let Provenance::Adaptive(score) = scenes[0].provenance_ref() {
    assert!(
      score.score() > 20.0,
      "the boundary score, not the quiet trailing frame's: {}",
      score.score()
    );
  } else {
    panic!(
      "expected an adaptive boundary: {:?}",
      scenes[0].provenance_ref()
    );
  }
}

#[test]
fn oversized_histogram_bins_are_rejected_at_construction() {
  // The bin count drives three eager per-lane buffer allocations: a
  // hostile count must surface as OptionsError instead of reaching the
  // allocator.
  let options = Options::new().with_luma_histogram(
    histogram::Options::new().with_bins(NonZeroUsize::new(usize::MAX).expect("nz")),
  );
  let Err(err) = Detector::try_new(options) else {
    panic!("oversized bins must be rejected");
  };
  assert!(
    matches!(err, OptionsError::HistogramBinsTooLarge(_)),
    "{err:?}"
  );
}

#[test]
fn same_push_deferral_survives_finalize() {
  // The same contention as above, but the stream ENDS on the
  // contention push: finalize must drain the deferred report before
  // the trailing close instead of swallowing the boundary.
  let options = Options::new()
    .with_detectors(Detectors::LUMA_HISTOGRAM | Detectors::ADAPTIVE)
    .with_luma_histogram(histogram::Options::new().with_min_duration(Duration::ZERO))
    .with_adaptive(
      adaptive::Options::new()
        .with_window_width(2)
        .with_min_duration(Duration::ZERO),
    )
    .with_select(select::Options::new().with_target_interval(Duration::from_millis(100)));
  let mut det = detector(options);
  let a = Fixture::solid(120, 10, 200);
  let b = Fixture::solid(120, 138, 200);
  let c = Fixture::solid(180, 138, 200);

  let mut outputs = push_run(&mut det, &a, 0, 20, 40);
  outputs.extend(push_run(&mut det, &b, 800, 2, 40));
  // The single contention push: the adaptive lane surfaces the 800 ms
  // hue cut, the luma histogram reports 880 ms — and the stream ends.
  outputs.extend(push_run(&mut det, &c, 880, 1, 40));
  outputs.extend(det.finalize());

  let scenes = scenes(&outputs);
  assert_eq!(
    scenes.len(),
    3,
    "the end-of-stream deferred boundary still closes: {:?}",
    scenes.iter().map(|s| s.range()).collect::<Vec<_>>()
  );
  assert_eq!(scenes[0].range().end().pts(), 800);
  assert_eq!(scenes[1].range().end().pts(), 880);
  assert!(scenes[1].provenance_ref().is_histogram());
  assert_eq!(scenes[2].range().end().pts(), 881);
}

#[test]
fn cutless_stream_honors_max_scene_duration() {
  // A cutless stream must still be force-split: the cap contends on
  // every push as the lowest-precedence candidate.
  let options = Options::new()
    .with_detectors(Detectors::all())
    .with_max_scene_duration(Some(Duration::from_millis(200)))
    .with_select(select::Options::new().with_target_interval(Duration::from_millis(100)));
  let mut det = detector(options);
  let fix = Fixture::striped(40, 220, 20, 180);
  let mut outputs = push_run(&mut det, &fix, 0, 30, 33);
  outputs.extend(det.finalize());

  let scenes = scenes(&outputs);
  assert!(
    scenes.len() >= 3,
    "a 200 ms cap over ~1 s of cut-less content forces splits, got {}",
    scenes.len()
  );
  assert!(scenes[0].provenance_ref().is_max_span());
}

#[test]
fn max_scene_duration_is_honored_at_exact_cap_boundaries() {
  // Whole-second frames against the default 15 s cap: the frame AT the
  // cap forces the split, so no emitted range exceeds the configured
  // maximum even in a coarse canonical timebase.
  let options = Options::new()
    .with_detectors(Detectors::THRESHOLD)
    .with_select(select::Options::new().with_target_interval(Duration::from_secs(1)));
  let mut det = detector(options);
  let fix = Fixture::solid(120, 10, 10);
  let coarse = Timebase::new(1, NonZeroU32::new(1).expect("nz"));

  let mut outputs = Vec::new();
  for s in 0..=15_i64 {
    if let Some(o) = det
      .push(fix.frames(Timestamp::new(s, coarse)))
      .expect("ordered stream")
    {
      outputs.push(o);
    }
  }
  outputs.extend(det.finalize());

  let scenes = scenes(&outputs);
  assert_eq!(
    scenes.len(),
    2,
    "exact-cap split plus the one-tick residue: {:?}",
    scenes.iter().map(|s| s.range()).collect::<Vec<_>>()
  );
  assert!(scenes[0].provenance_ref().is_max_span());
  assert_eq!(scenes[0].range().end().pts(), 15);
  assert_eq!(scenes[1].range().end().pts(), 16);
}

#[cfg(feature = "serde")]
#[test]
fn options_round_trip_through_serde() {
  // The crate contract: every Options type serializes under the serde
  // feature. Bitflags travel as their human-readable flag strings.
  let options = Options::new()
    .with_detectors(Detectors::default() - Detectors::PHASH)
    .with_max_scene_duration(Some(Duration::from_secs(7)));
  let json = serde_json::to_string(&options).expect("serializes");
  let back: Options = serde_json::from_str(&json).expect("deserializes");
  assert_eq!(back.detectors(), options.detectors());
  assert_eq!(back.extractors(), options.extractors());
  assert_eq!(back.max_scene_duration(), options.max_scene_duration());
}

#[test]
fn out_of_order_frames_are_rejected_without_state_damage() {
  // A regressing frame must be refused with state untouched: every
  // downstream invariant (lane recency, window finalize anchoring,
  // canonical ranges) assumes a non-decreasing stream.
  let mut det = detector(fixture_options());
  let fix = Fixture::solid(120, 10, 10);
  let mut outputs = Vec::new();
  for ms in [0_i64, 500, 1000] {
    if let Some(o) = det.push(fix.frames(ts(ms))).expect("ordered stream") {
      outputs.push(o);
    }
  }
  assert_eq!(
    det.push(fix.frames(ts(700))).expect_err("regressing frame"),
    PushError::OutOfOrder
  );
  // Equal timestamps stay accepted, and the stream continues past the
  // rejected frame as if it never arrived.
  let _ = det.push(fix.frames(ts(1000))).expect("equal timestamp");
  outputs.extend(det.finalize());
  let scenes = scenes(&outputs);
  assert_eq!(scenes.len(), 1);
  assert_eq!(
    scenes[0].range().end().pts(),
    1001,
    "finalization still covers the latest frame: {:?}",
    scenes.iter().map(|s| s.range()).collect::<Vec<_>>()
  );
}

#[test]
fn oversized_phash_working_size_is_rejected_at_construction() {
  // size * lowpass drives quadratic allocations: a hostile working
  // size must surface as OptionsError instead of reaching the
  // allocator.
  let options = Options::new().with_phash(phash::Options::new().with_size(65_536).with_lowpass(1));
  let Err(err) = Detector::try_new(options) else {
    panic!("oversized phash working size must be rejected");
  };
  assert!(matches!(err, OptionsError::PhashSizeTooLarge(_)), "{err:?}");
}

#[test]
fn max_pts_frame_is_rejected() {
  // A frame at the representable pts limit can never be closed past;
  // it is refused with state untouched rather than silently dropped
  // from the final scene.
  let mut det = detector(fixture_options());
  let fix = Fixture::solid(120, 10, 10);
  let _ = det.push(fix.frames(ts(0))).expect("ordered stream");
  assert_eq!(
    det.push(fix.frames(ts(i64::MAX))).expect_err("limit frame"),
    PushError::TimestampAtLimit
  );
  let _ = det.push(fix.frames(ts(40))).expect("stream continues");
}

#[test]
fn canonical_unrepresentable_close_is_rejected() {
  // The canonical timebase is the FIRST frame's: a later coarse frame
  // whose one-past close overflows the fine canonical pts must be
  // refused — the rescale sibling of the raw-limit guard.
  let options = Options::new()
    .with_detectors(Detectors::LUMA_HISTOGRAM | Detectors::CONTENT)
    .with_select(select::Options::new().with_target_interval(Duration::from_millis(100)));
  let mut det = detector(options);
  let fix = Fixture::solid(120, 10, 10);
  let micros = Timebase::new(1, NonZeroU32::new(1_000_000).expect("nz"));
  let coarse = Timebase::new(1, NonZeroU32::new(1).expect("nz"));
  let _ = det
    .push(fix.frames(Timestamp::new(0, micros)))
    .expect("first frame sets the canonical timebase");
  // 9.3e12 seconds is 9.3e18 microseconds — past i64::MAX.
  assert_eq!(
    det
      .push(fix.frames(Timestamp::new(9_300_000_000_000, coarse)))
      .expect_err("unrepresentable close"),
    PushError::TimestampAtLimit
  );
  // A representable later frame still streams.
  let _ = det
    .push(fix.frames(Timestamp::new(40_000, micros)))
    .expect("stream continues");
}

#[test]
fn oversized_edge_kernel_is_rejected_at_construction() {
  // An explicit huge kernel with a nonzero edge weight is a first-push
  // CPU hazard: rejected as a typed error for both lanes.
  let hostile = content::Options::new()
    .with_weights(content::Components::new(1.0, 1.0, 1.0, 1.0))
    .with_kernel_size(Some(u32::MAX));
  let Err(err) = Detector::try_new(Options::new().with_content(hostile)) else {
    panic!("oversized content kernel must be rejected");
  };
  assert!(matches!(err, OptionsError::KernelTooLarge(_)), "{err:?}");
}

#[test]
fn lane_rescale_cannot_saturate_under_mixed_timebases() {
  // A nanosecond-timebase fade-out followed by a coarse fade-in whose
  // value is beyond the fine timebase's i64 range: lanes only ever see
  // canonically-normalized timestamps, so the threshold fade
  // interpolation cannot saturate and the cut stays finite and sane.
  let options = Options::new()
    .with_detectors(Detectors::THRESHOLD)
    .with_threshold(threshold::Options::new().with_min_duration(Duration::ZERO))
    .with_max_scene_duration(None)
    .with_select(select::Options::new().with_target_interval(Duration::from_secs(1)));
  let mut det = detector(options);
  let bright = Fixture::solid(160, 10, 10);
  let black = Fixture::solid(4, 10, 10);
  let coarse = Timebase::new(1, NonZeroU32::new(1).expect("nz"));
  let nanos = Timebase::new(1, NonZeroU32::new(1_000_000_000).expect("nz"));

  let mut outputs = Vec::new();
  for (fix, t) in [
    (&bright, Timestamp::new(0, coarse)),
    (&bright, Timestamp::new(1, coarse)),
    (&black, Timestamp::new(3_000_000_000, nanos)),
    (&black, Timestamp::new(4_000_000_000, nanos)),
    (&bright, Timestamp::new(9_300_000_000, coarse)),
    (&bright, Timestamp::new(9_300_000_001, coarse)),
  ] {
    if let Some(o) = det.push(fix.frames(t)).expect("ordered, representable") {
      outputs.push(o);
    }
  }
  outputs.extend(det.finalize());

  let scenes = scenes(&outputs);
  assert!(
    scenes.len() >= 2,
    "the fade closed: {:?}",
    scenes.iter().map(|s| s.range()).collect::<Vec<_>>()
  );
  let cut = scenes[0].range().end().pts();
  assert!(
    cut > 4 && cut < 9_300_000_000,
    "a finite interpolated cut, not a saturated one: {cut}"
  );
}

#[test]
fn sparse_one_frame_windows_still_emit() {
  // One frame per window: with the selector's first/last-bucket
  // margins disabled for cascade windows, the lone frame in each
  // window is scored and emitted — a non-empty window never silently
  // drops its only frame (the R36 sparse-window fix).
  let options = Options::new()
    .with_detectors(Detectors::CONTENT)
    .with_select(select::Options::new().with_target_interval(Duration::from_secs(1)));
  let mut det = detector(options);
  let fix = Fixture::striped(40, 220, 20, 180);
  let mut outputs = Vec::new();
  for s in 0..6_i64 {
    if let Some(o) = det.push(fix.frames(ts(s * 1000))).expect("ordered") {
      outputs.push(o);
    }
  }
  outputs.extend(det.finalize());
  let kfs = keyframes(&outputs);
  assert!(
    kfs.len() >= 5,
    "each sparse one-frame window emits its frame, got {}",
    kfs.len()
  );
}

#[test]
fn lag_holdback_keeps_post_cut_frames_for_the_new_scene() {
  // The adaptive lag (window 8 → several hundred ms) far exceeds the
  // 200 ms keyframe window. Without the lag hold-back, periodic
  // finalize drains the new scene's leading frames into a window
  // straddling the not-yet-confirmed cut, losing them — the new
  // scene's first keyframe then slips a full window past the cut. The
  // hold-back retains those frames until the cut confirms, so the new
  // scene is sampled from its own first frame (the R36 lagged-drain
  // fix; without it `first_new` is one window later).
  let options = Options::new()
    .with_detectors(Detectors::ADAPTIVE)
    .with_adaptive(adaptive::Options::new().with_window_width(8))
    .with_select(select::Options::new().with_target_interval(Duration::from_millis(200)));
  let mut det = detector(options);
  let red = Fixture::solid(60, 10, 200);
  let blue = Fixture::solid(180, 120, 200);
  let mut outputs = push_run(&mut det, &red, 0, 33, 40); // 0..1320
  outputs.extend(push_run(&mut det, &blue, 1320, 40, 40)); // 1320..
  outputs.extend(det.finalize());

  let scenes = scenes(&outputs);
  assert_eq!(
    scenes.len(),
    2,
    "the lagged cut splits: {:?}",
    scenes.iter().map(|s| s.range()).collect::<Vec<_>>()
  );
  let cut = scenes[0].range().end();
  let first_new = keyframes(&outputs)
    .iter()
    .map(|k| k.timestamp())
    .filter(|t| t.cmp_semantic(&cut).is_ge())
    .min_by(|a, b| a.cmp_semantic(b));
  assert!(
    first_new.is_some_and(|t| t.cmp_semantic(&cut).is_eq()),
    "the new scene's leading frame is retained and sampled: cut={:?} first_new={:?}",
    cut,
    first_new
  );
}

#[test]
fn keyframes_partition_a_multi_scene_stream() {
  // The load-bearing association contract: across a multi-scene
  // stream, every emitted keyframe falls in EXACTLY ONE scene's
  // half-open range, and the ranges tile the stream (pairwise
  // adjacent, no gap, no overlap).
  let mut det = detector(fixture_options());
  let a = Fixture::solid(60, 10, 200);
  let b = Fixture::solid(180, 120, 200);
  let c = Fixture::solid(110, 60, 80);
  let mut outputs = push_run(&mut det, &a, 0, 30, 40);
  outputs.extend(push_run(&mut det, &b, 1200, 30, 40));
  outputs.extend(push_run(&mut det, &c, 2400, 30, 40));
  outputs.extend(det.finalize());

  let scenes = scenes(&outputs);
  assert!(
    scenes.len() >= 3,
    "multi-scene stream: {:?}",
    scenes.iter().map(|s| s.range()).collect::<Vec<_>>()
  );
  for pair in scenes.windows(2) {
    assert!(
      pair[0]
        .range()
        .end()
        .cmp_semantic(&pair[1].range().start())
        .is_eq(),
      "scene ranges tile (adjacent): {:?} then {:?}",
      pair[0].range(),
      pair[1].range()
    );
  }
  for k in keyframes(&outputs) {
    let owners = scenes
      .iter()
      .filter(|sc| {
        k.timestamp().cmp_semantic(&sc.range().start()).is_ge()
          && k.timestamp().cmp_semantic(&sc.range().end()).is_lt()
      })
      .count();
    assert_eq!(
      owners,
      1,
      "keyframe {:?} owned by exactly one scene, got {}",
      k.timestamp(),
      owners
    );
  }
}

#[test]
fn adaptive_at_cut_tail_drains_per_window() {
  // Under adaptive lag the periodic finalize holds back the scene's
  // last frames; they drain at the cut as the held-back tail, which
  // must sub-bucket into one keyframe per window — not collapse to a
  // single keyframe. A ~1600 ms first scene at a 200 ms window yields
  // several keyframes, not ~1.
  let options = Options::new()
    .with_detectors(Detectors::ADAPTIVE)
    .with_adaptive(adaptive::Options::new().with_window_width(8))
    .with_select(select::Options::new().with_target_interval(Duration::from_millis(200)));
  let mut det = detector(options);
  let red = Fixture::solid(60, 10, 200);
  let blue = Fixture::solid(180, 120, 200);
  let mut outputs = push_run(&mut det, &red, 0, 40, 40); // 0..1600
  outputs.extend(push_run(&mut det, &blue, 1600, 40, 40));
  outputs.extend(det.finalize());

  let scenes = scenes(&outputs);
  assert_eq!(scenes.len(), 2, "the lagged cut splits");
  let cut = scenes[0].range().end();
  let first_scene_kfs = keyframes(&outputs)
    .iter()
    .filter(|k| k.timestamp().cmp_semantic(&cut).is_lt())
    .count();
  assert!(
    first_scene_kfs >= 6,
    "the held-back tail drains per window (≈8), not collapsed: got {first_scene_kfs}"
  );
}

#[test]
fn quality_and_fallback_keyframes_can_both_occur() {
  // Strict (Quality) vs fallback selection is decided per window. A
  // stream alternating textured (gate-passing) and flat (gate-failing)
  // frames within one bright scene must produce BOTH provenances.
  let options = Options::new()
    .with_detectors(Detectors::THRESHOLD) // bright frames never fade → one scene
    .with_select(select::Options::new().with_target_interval(Duration::from_millis(200)));
  let mut det = detector(options);
  let sharp = Fixture::striped(150, 230, 20, 180);
  let flat = Fixture::solid(190, 10, 10);
  let mut outputs = Vec::new();
  for i in 0..40_i64 {
    let fix = if (i / 5) % 2 == 0 { &sharp } else { &flat };
    if let Some(o) = det.push(fix.frames(ts(i * 40))).expect("ordered") {
      outputs.push(o);
    }
  }
  outputs.extend(det.finalize());
  let kfs = keyframes(&outputs);
  let quality = kfs
    .iter()
    .filter(|k| k.provenance_ref().is_quality())
    .count();
  let fallback = kfs
    .iter()
    .filter(|k| k.provenance_ref().is_fallback())
    .count();
  assert!(
    quality >= 1 && fallback >= 1,
    "both selection paths occur: quality={quality} fallback={fallback}"
  );
}

#[test]
#[ignore = "heavy: overfills the 65_536-entry selector buffer (~66k pushes); run with --ignored"]
fn buffer_overflow_keeps_the_stream_well_formed() {
  // BufferFull is reachable with a huge keyframe window + no forced
  // split: frames accumulate unflushed past the selector's cap. The
  // overflow recovery drains only up to `safe_horizon` (the same
  // hold-back the periodic path uses — never past it). This is a
  // robustness reproduction: the overflow path must not panic, the
  // stream must still split, and the post-overflow scene must still be
  // sampled (no wholesale keyframe loss). Identical frames fill the
  // buffer; then a transition cuts.
  let options = Options::new()
    .with_detectors(Detectors::ADAPTIVE)
    .with_adaptive(adaptive::Options::new().with_window_width(2))
    .with_max_scene_duration(None)
    .with_select(select::Options::new().with_target_interval(Duration::from_secs(36_000)));
  let mut det = detector(options);
  let red = Fixture::solid(60, 10, 200);
  let blue = Fixture::solid(180, 120, 200);
  // Overfill the 65_536-entry selector buffer with one un-cut run.
  let mut outputs = Vec::new();
  for i in 0..66_000_i64 {
    if let Some(o) = det.push(red.frames(ts(i * 10))).expect("ordered") {
      outputs.push(o);
    }
  }
  let flip = 66_000_i64 * 10;
  for i in 0..30_i64 {
    if let Some(o) = det.push(blue.frames(ts(flip + i * 10))).expect("ordered") {
      outputs.push(o);
    }
  }
  outputs.extend(det.finalize());

  let scenes = scenes(&outputs);
  assert!(
    scenes.len() >= 2,
    "the transition still splits despite overflow"
  );
  let cut = scenes[scenes.len() - 2].range().end();
  // The new scene is sampled from its own leading frames — the
  // overflow drain did not consume the lag-reclaimable tail.
  let first_new = keyframes(&outputs)
    .iter()
    .map(|k| k.timestamp())
    .filter(|t| t.cmp_semantic(&cut).is_ge())
    .min_by(|a, b| a.cmp_semantic(b));
  assert!(
    first_new.is_some_and(|t| t
      .duration_since(&cut)
      .is_some_and(|d| d < Duration::from_millis(200))),
    "new scene sampled near its start after overflow: cut={cut:?} first_new={first_new:?}"
  );
}

#[test]
fn sparse_timestamp_jump_keeps_push_bounded() {
  // A tiny target_interval plus a large timestamp gap must NOT make one
  // push walk a window per empty interval (a CPU-DoS). The catch-up is
  // a single bounded finalize (finalize_shot clamps the bucket count),
  // so this returns promptly with a bounded keyframe count rather than
  // spinning ~14 million iterations.
  let ns = Timebase::new(1, NonZeroU32::new(1_000_000_000).expect("nz"));
  let options = Options::new()
    .with_detectors(Detectors::CONTENT)
    .with_max_scene_duration(None)
    .with_select(select::Options::new().with_target_interval(Duration::from_micros(1)));
  let mut det = detector(options);
  let fix = Fixture::striped(40, 220, 20, 180);
  let _ = det
    .push(fix.frames(Timestamp::new(0, ns)))
    .expect("ordered");
  let _ = det
    .push(fix.frames(Timestamp::new(14_000_000_000, ns)))
    .expect("ordered");
  let mut outputs = Vec::new();
  outputs.extend(det.finalize());
  let kfs = keyframes(&outputs);
  assert!(
    kfs.len() <= 64,
    "catch-up is bounded, not one finalize per empty window: {}",
    kfs.len()
  );
}
