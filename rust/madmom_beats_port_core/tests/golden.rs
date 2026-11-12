// SPDX-License-Identifier: BSD-2-Clause

use madmom_beats_port_core::{analyze, CoreConfig};
use serde::Deserialize;
use std::path::PathBuf;

// The goldens are madmom's results on macOS arm64. The port computes the same
// bits on every platform, so it differs from them in the last bits of some
// activations, which moves a few beat times by up to about 0.14 ms.
const TIME_TOLERANCE_SEC: f32 = 0.001;
const CONFIDENCE_TOLERANCE: f32 = 0.005;

#[derive(Debug, Deserialize)]
struct GoldenEvents {
    beats: Vec<GoldenBeat>,
}

#[derive(Debug, Deserialize)]
struct GoldenBeat {
    time_sec: f32,
    beat_in_bar: u32,
    confidence: f32,
}

fn fixtures_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("fixtures")
}

fn repo_root() -> PathBuf {
    fixtures_root().parent().unwrap().to_path_buf()
}

/// Mono samples as madmom loads the wav, and whether they are 16-bit PCM
/// (`int16_scaling`). madmom keeps a 16-bit signal as integers and downmixes
/// it to the channel mean truncated to an integer; float samples are
/// averaged as floats.
fn load_wav_mono(path: &PathBuf) -> (Vec<f32>, bool) {
    let mut reader = hound::WavReader::open(path).expect("failed to open wav");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 44_100, "unexpected sample rate");
    let channels = spec.channels as usize;
    match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => {
            let raw: Vec<i32> = reader.samples::<i16>().map(|s| s.unwrap() as i32).collect();
            let samples = raw
                .chunks_exact(channels)
                .map(|frame| {
                    let mean = frame.iter().sum::<i32>() as f64 / channels as f64;
                    mean.trunc() as f32 / 32768.0
                })
                .collect();
            (samples, true)
        }
        (hound::SampleFormat::Float, 32) => {
            let raw: Vec<f32> = reader.samples::<f32>().map(|s| s.unwrap()).collect();
            let samples = raw
                .chunks_exact(channels)
                .map(|frame| frame.iter().sum::<f32>() / channels as f32)
                .collect();
            (samples, false)
        }
        (format, bits) => panic!("unsupported wav format {format:?} {bits}-bit"),
    }
}

fn load_events(path: &PathBuf) -> GoldenEvents {
    let data = std::fs::read_to_string(path).expect("events.json missing");
    serde_json::from_str(&data).expect("invalid events.json")
}

fn assert_close(expected: &[f32], actual: &[f32], tolerance: f32, label: &str) {
    assert_eq!(expected.len(), actual.len(), "{label} count mismatch");
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        assert!(
            (e - a).abs() <= tolerance,
            "{label} {i}: expected {e}, got {a}"
        );
    }
}

#[test]
fn golden_parity_all() {
    let fixtures = fixtures_root();
    let golden_root = fixtures.join("golden");
    let audio_root = fixtures.join("audio");
    let mut config = CoreConfig::default();
    let models_root = repo_root().join("models");
    config.model.model_json = models_root
        .join("downbeats_blstm.json")
        .to_string_lossy()
        .to_string();
    config.model.weights_npz = models_root
        .join("downbeats_blstm_weights.npz")
        .to_string_lossy()
        .to_string();

    let filter = std::env::var("FIXTURE").ok();

    for entry in std::fs::read_dir(&golden_root).expect("golden dir missing") {
        let entry = entry.expect("bad dir entry");
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            continue;
        }
        let file_type = entry.file_type().expect("bad dir entry type");
        if !file_type.is_dir() {
            continue;
        }
        if let Some(ref wanted) = filter {
            if name != *wanted {
                continue;
            }
        }
        let golden_dir = golden_root.join(&name);
        let audio_path = audio_root.join(format!("{}.wav", name));
        let events_path = golden_dir.join("events.json");

        println!("fixture {}: loading audio", name);
        let (samples, int16_scaling) = load_wav_mono(&audio_path);
        config.feature.int16_scaling = int16_scaling;
        let duration_sec = samples.len() as f32 / config.feature.sample_rate as f32;
        println!(
            "fixture {}: analyzing ({} samples, {:.1}s)",
            name,
            samples.len(),
            duration_sec
        );
        let t0 = std::time::Instant::now();
        let output = analyze(&samples, config.feature.sample_rate, &config)
            .unwrap_or_else(|e| panic!("analysis failed for {}: {}", name, e));
        println!("fixture {}: analyze done in {:.2?}", name, t0.elapsed());
        assert_eq!(
            output.fps,
            config.feature.fps.round() as u32,
            "fps mismatch"
        );

        println!("fixture {}: loading goldens", name);
        let events = load_events(&events_path);
        let beat_times_gold: Vec<f32> = events.beats.iter().map(|b| b.time_sec).collect();
        assert_close(
            &beat_times_gold,
            &output.beat_times,
            TIME_TOLERANCE_SEC,
            "beat time",
        );

        let beat_numbers_gold: Vec<u32> = events.beats.iter().map(|b| b.beat_in_bar).collect();
        assert_eq!(
            beat_numbers_gold, output.beat_numbers,
            "beats beat_in_bar mismatch for {}",
            name
        );

        let beat_conf_gold: Vec<f32> = events.beats.iter().map(|b| b.confidence).collect();
        assert_close(
            &beat_conf_gold,
            &output.beat_confidences,
            CONFIDENCE_TOLERANCE,
            "confidence",
        );

        assert_eq!(
            output.beat_times.len(),
            output.beat_numbers.len(),
            "beat_times/beat_numbers length mismatch for {}",
            name
        );
        assert_eq!(
            output.beat_times.len(),
            output.beat_confidences.len(),
            "beat_times/beat_confidences length mismatch for {}",
            name
        );

        for i in 1..output.beat_times.len() {
            assert!(
                output.beat_times[i] > output.beat_times[i - 1],
                "non-increasing beat_times at index {} for {}",
                i,
                name
            );
        }
        for beat_number in &output.beat_numbers {
            assert!(
                *beat_number >= 1,
                "invalid beat_number {} in {}",
                beat_number,
                name
            );
        }
        for confidence in &output.beat_confidences {
            assert!(
                (0.0..=1.0).contains(confidence),
                "invalid confidence {} in {}",
                confidence,
                name
            );
        }
        println!("fixture {}: parity ok", name);
    }
}
