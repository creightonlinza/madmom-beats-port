// SPDX-License-Identifier: BSD-2-Clause

//! Full-pipeline tests on generated audio, so they need no local fixtures.
//!
//! The snapshots in `tests/data/` hold the expected results. Set
//! `UPDATE_SNAPSHOTS=1` to rewrite them, only for an intended change to results
//! that `golden_parity_all` also accepts.

use madmom_beats_port_core::{
    analyze, analyze_with_model_data, decode_network_outputs, extract_features, feature_columns,
    AnalysisOutput, BeatModel, CoreConfig, RhythmError,
};
use std::path::PathBuf;

const SAMPLE_RATE: u32 = 44_100;
const BEAT_INTERVAL_SEC: f32 = 0.5;

// Wide enough for floating-point differences between platforms, which reach
// about 16 ms and 0.07 between native and WASM builds.
const TIME_TOLERANCE_SEC: f32 = 0.02;
const CONFIDENCE_TOLERANCE: f32 = 0.1;

/// Deterministic click track: decaying sine bursts, with every bar's first
/// beat louder and lower.
fn click_track(seconds: usize, interval_sec: f32, beats_per_bar: usize) -> Vec<f32> {
    const BURST_SAMPLES: usize = 4_410;
    let sample_rate = SAMPLE_RATE as usize;
    let mut samples = vec![0.0f32; seconds * sample_rate];
    let interval = (interval_sec * SAMPLE_RATE as f32) as usize;
    let beats = samples.len() / interval;
    for beat in 0..beats {
        let downbeat = beat % beats_per_bar == 0;
        let (amplitude, frequency) = if downbeat { (0.9, 220.0) } else { (0.5, 880.0) };
        let start = beat * interval;
        for i in 0..BURST_SAMPLES.min(samples.len() - start) {
            let t = i as f32 / SAMPLE_RATE as f32;
            let envelope = (-t * 40.0).exp();
            samples[start + i] +=
                amplitude * envelope * (2.0 * std::f32::consts::PI * frequency * t).sin();
        }
    }
    samples
}

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../models")
        .canonicalize()
        .expect("models directory missing")
}

fn config() -> CoreConfig {
    let mut config = CoreConfig::default();
    let models = models_dir();
    config.model.model_json = models
        .join("downbeats_blstm.json")
        .to_string_lossy()
        .to_string();
    config.model.weights_npz = models
        .join("downbeats_blstm_weights.npz")
        .to_string_lossy()
        .to_string();
    config
}

fn model_data() -> (String, Vec<u8>) {
    let models = models_dir();
    (
        std::fs::read_to_string(models.join("downbeats_blstm.json")).unwrap(),
        std::fs::read(models.join("downbeats_blstm_weights.npz")).unwrap(),
    )
}

fn assert_bits_equal(a: &AnalysisOutput, b: &AnalysisOutput) {
    assert_eq!(a.fps, b.fps);
    assert_eq!(a.beat_numbers, b.beat_numbers);
    let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&a.beat_times), bits(&b.beat_times));
    assert_eq!(bits(&a.beat_confidences), bits(&b.beat_confidences));
}

fn assert_matches_snapshot(name: &str, actual: &AnalysisOutput) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(format!("{name}.json"));
    if std::env::var("UPDATE_SNAPSHOTS").ok().as_deref() == Some("1") {
        let json = serde_json::to_string_pretty(actual).unwrap() + "\n";
        std::fs::write(&path, json).unwrap();
        return;
    }
    let expected: AnalysisOutput =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(expected.fps, actual.fps);
    assert_eq!(expected.beat_numbers, actual.beat_numbers);
    for (i, (e, a)) in expected
        .beat_times
        .iter()
        .zip(&actual.beat_times)
        .enumerate()
    {
        assert!((e - a).abs() <= TIME_TOLERANCE_SEC, "beat {i}: {e} vs {a}");
    }
    for (i, (e, a)) in expected
        .beat_confidences
        .iter()
        .zip(&actual.beat_confidences)
        .enumerate()
    {
        assert!(
            (e - a).abs() <= CONFIDENCE_TOLERANCE,
            "confidence {i}: {e} vs {a}"
        );
    }
}

/// Beats land on the clicks, and the bar position counts up from each louder
/// click.
fn assert_follows_click_track(output: &AnalysisOutput, beats_per_bar: u32) {
    assert!(!output.beat_times.is_empty());
    for (time, &number) in output.beat_times.iter().zip(&output.beat_numbers) {
        let beat = (time / BEAT_INTERVAL_SEC).round();
        assert!(
            (time - beat * BEAT_INTERVAL_SEC).abs() <= TIME_TOLERANCE_SEC,
            "beat at {time} is off the click grid"
        );
        let expected = (beat as u32) % beats_per_bar + 1;
        assert_eq!(number, expected, "wrong bar position at {time}");
    }
}

#[test]
fn four_four_click_track_matches_snapshot() {
    let samples = click_track(20, BEAT_INTERVAL_SEC, 4);
    let output = analyze(&samples, SAMPLE_RATE, &config()).unwrap();
    assert_follows_click_track(&output, 4);
    assert_matches_snapshot("click_4_4", &output);
}

#[test]
fn three_four_click_track_matches_snapshot() {
    let samples = click_track(20, BEAT_INTERVAL_SEC, 3);
    let output = analyze(&samples, SAMPLE_RATE, &config()).unwrap();
    assert_follows_click_track(&output, 3);
    assert_matches_snapshot("click_3_4", &output);
}

#[test]
fn model_data_entry_point_matches_file_entry_point() {
    let samples = click_track(10, BEAT_INTERVAL_SEC, 4);
    let config = config();
    let (model_json, weights) = model_data();
    let from_files = analyze(&samples, SAMPLE_RATE, &config).unwrap();
    let from_data =
        analyze_with_model_data(&samples, SAMPLE_RATE, &config, &model_json, &weights).unwrap();
    assert_bits_equal(&from_files, &from_data);
}

#[test]
fn staged_analysis_matches_single_call_bit_for_bit() {
    let samples = click_track(10, BEAT_INTERVAL_SEC, 3);
    let config = config();
    let (model_json, weights) = model_data();
    let single =
        analyze_with_model_data(&samples, SAMPLE_RATE, &config, &model_json, &weights).unwrap();

    let features = extract_features(&samples, SAMPLE_RATE, &config).unwrap();
    assert_eq!(features.ncols(), feature_columns(&config).unwrap());
    let model = BeatModel::from_data(&model_json, &weights).unwrap();
    // Networks finish in any order; the outputs are decoded in network order.
    let mut outputs: Vec<_> = (0..model.num_networks())
        .rev()
        .map(|index| (index, model.run_network(&features, index).unwrap()))
        .collect();
    outputs.sort_by_key(|(index, _)| *index);
    let outputs: Vec<_> = outputs.into_iter().map(|(_, output)| output).collect();

    let (combined, combined_logp) = decode_network_outputs(outputs.clone(), &config, None)
        .unwrap()
        .expect("a bar length should fit");
    assert_bits_equal(&single, &combined);

    let mut best: Option<(AnalysisOutput, f64)> = None;
    for &beats_per_bar in &config.dbn.beats_per_bar {
        let mut one_bar_length = config.clone();
        one_bar_length.dbn.beats_per_bar = vec![beats_per_bar];
        let decoded = decode_network_outputs(outputs.clone(), &one_bar_length, None).unwrap();
        if let Some((output, logp)) = decoded {
            if best.as_ref().is_none_or(|(_, best_logp)| logp > *best_logp) {
                best = Some((output, logp));
            }
        }
    }
    let (separate, separate_logp) = best.expect("a bar length should fit");
    assert_bits_equal(&single, &separate);
    assert_eq!(combined_logp.to_bits(), separate_logp.to_bits());
}

#[test]
fn silence_has_no_beats() {
    let samples = vec![0.0f32; SAMPLE_RATE as usize * 5];
    let output = analyze(&samples, SAMPLE_RATE, &config()).unwrap();
    assert!(output.beat_times.is_empty());
    assert!(output.beat_numbers.is_empty());
    assert!(output.beat_confidences.is_empty());
}

#[test]
fn invalid_input_is_rejected_by_every_entry_point() {
    let config = config();
    let (model_json, weights) = model_data();
    let samples = vec![0.0f32; 1_000];
    let wrong_rate = SAMPLE_RATE / 2;

    let results = [
        analyze(&samples, wrong_rate, &config).err(),
        analyze_with_model_data(&samples, wrong_rate, &config, &model_json, &weights).err(),
        extract_features(&samples, wrong_rate, &config).err(),
        analyze(&[], SAMPLE_RATE, &config).err(),
        extract_features(&[], SAMPLE_RATE, &config).err(),
    ];
    for result in results {
        assert!(matches!(result, Some(RhythmError::InvalidInput(_))));
    }

    let mut invalid = config.clone();
    invalid.dbn.num_tempi = 0;
    assert!(matches!(
        extract_features(&samples, SAMPLE_RATE, &invalid),
        Err(RhythmError::InvalidInput(_))
    ));
    assert!(matches!(
        feature_columns(&invalid),
        Err(RhythmError::InvalidInput(_))
    ));
}

#[test]
fn staged_network_rejects_an_unknown_index() {
    let (model_json, weights) = model_data();
    let config = config();
    let samples = click_track(1, BEAT_INTERVAL_SEC, 4);
    let features = extract_features(&samples, SAMPLE_RATE, &config).unwrap();
    let model = BeatModel::from_data(&model_json, &weights).unwrap();
    assert!(model.run_network(&features, model.num_networks()).is_err());
}
