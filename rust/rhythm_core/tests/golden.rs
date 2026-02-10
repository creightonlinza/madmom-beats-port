use rhythm_core::{analyze, CoreConfig};
use serde::Deserialize;
use std::path::PathBuf;

const ACTIVATION_TOLERANCE: f32 = 1e-3;
const FRAME_TOLERANCE_SEC: f32 = 0.005;

#[derive(Debug, Deserialize)]
struct GoldenEvents {
    beats: Vec<GoldenBeat>,
    downbeats: Vec<GoldenDownbeat>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct GoldenBeat {
    time_sec: f32,
    confidence: f32,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct GoldenDownbeat {
    time_sec: f32,
    beat_in_bar: usize,
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

fn load_wav_mono(path: &PathBuf) -> (Vec<f32>, u32) {
    let mut reader = hound::WavReader::open(path).expect("failed to open wav");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 44_100, "unexpected sample rate");
    let channels = spec.channels as usize;
    let mut samples = Vec::new();

    match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            let mut frame = Vec::with_capacity(channels);
            for s in reader.samples::<i32>() {
                let v = s.expect("invalid sample") as f32 / max;
                frame.push(v);
                if frame.len() == channels {
                    let sum: f32 = frame.iter().sum();
                    samples.push(sum / channels as f32);
                    frame.clear();
                }
            }
        }
        hound::SampleFormat::Float => {
            let mut frame = Vec::with_capacity(channels);
            for s in reader.samples::<f32>() {
                let v = s.expect("invalid sample");
                frame.push(v);
                if frame.len() == channels {
                    let sum: f32 = frame.iter().sum();
                    samples.push(sum / channels as f32);
                    frame.clear();
                }
            }
        }
    }
    (samples, spec.sample_rate)
}

fn load_activation_arrays(path: &PathBuf) -> (Vec<f32>, Vec<f32>) {
    use ndarray::{Ix1, OwnedRepr};
    use ndarray_npy::NpzReader;
    use std::fs::File;
    let file = File::open(path).expect("activations.npz missing");
    let mut npz = NpzReader::new(file).expect("npz open failed");
    let beat = npz
        .by_name::<OwnedRepr<f32>, Ix1>("beat.npy")
        .expect("beat array missing")
        .to_vec();
    let downbeat = npz
        .by_name::<OwnedRepr<f32>, Ix1>("downbeat.npy")
        .expect("downbeat array missing")
        .to_vec();
    (beat, downbeat)
}

fn load_events(path: &PathBuf) -> GoldenEvents {
    let data = std::fs::read_to_string(path).expect("events.json missing");
    serde_json::from_str(&data).expect("invalid events.json")
}

fn compare_activations(expected: &[f32], actual: &[f32], label: &str) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "length mismatch for {}",
        label
    );
    let mut max_diff = 0.0f32;
    let len = expected.len();
    let progress_step = if len >= 200_000 { len / 10 } else { 0 };
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        if progress_step > 0 && i % progress_step == 0 {
            let pct = (i as f32 / len as f32) * 100.0;
            println!("activation compare {}: {:.0}%", label, pct);
        }
        let diff = (e - a).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    assert!(
        max_diff <= ACTIVATION_TOLERANCE,
        "max diff {} for {} exceeds tolerance {}",
        max_diff,
        label,
        ACTIVATION_TOLERANCE
    );
}

fn compare_event_times(expected: &[f32], actual: &[f32], label: &str) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "event count mismatch for {}",
        label
    );
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        let diff = (e - a).abs();
        assert!(
            diff <= FRAME_TOLERANCE_SEC,
            "event {} in {} differs by {}s",
            i,
            label,
            diff
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
        let activations_path = golden_dir.join("activations.npz");
        let events_path = golden_dir.join("events.json");

        println!("fixture {}: loading audio", name);
        let (samples, sample_rate) = load_wav_mono(&audio_path);
        let duration_sec = samples.len() as f32 / sample_rate as f32;
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

        println!("fixture {}: loading goldens", name);
        let (beat_gold, downbeat_gold) = load_activation_arrays(&activations_path);
        compare_activations(&beat_gold, &output.activations.beat, "beat");
        compare_activations(&downbeat_gold, &output.activations.downbeat, "downbeat");

        let events = load_events(&events_path);
        let beat_times_gold: Vec<f32> = events.beats.iter().map(|b| b.time_sec).collect();
        let beat_times: Vec<f32> = output.events.beats.iter().map(|b| b.time_sec).collect();
        compare_event_times(&beat_times_gold, &beat_times, "beats");

        let downbeat_times_gold: Vec<f32> = events.downbeats.iter().map(|b| b.time_sec).collect();
        let downbeat_times: Vec<f32> = output.events.downbeats.iter().map(|b| b.time_sec).collect();
        compare_event_times(&downbeat_times_gold, &downbeat_times, "downbeats");

        let downbeat_numbers_gold: Vec<usize> =
            events.downbeats.iter().map(|b| b.beat_in_bar).collect();
        let downbeat_numbers: Vec<usize> = output
            .events
            .downbeats
            .iter()
            .map(|b| b.beat_in_bar)
            .collect();
        assert_eq!(
            downbeat_numbers_gold, downbeat_numbers,
            "downbeat beat_in_bar mismatch for {}",
            name
        );
        println!("fixture {}: parity ok", name);
    }
}
