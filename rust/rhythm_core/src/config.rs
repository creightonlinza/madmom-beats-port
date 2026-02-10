use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    pub sample_rate: u32,
    pub fps: f32,
    pub frame_sizes: [usize; 3],
    pub num_bands: [usize; 3],
    pub fmin: f32,
    pub fmax: f32,
    pub diff_ratio: f32,
    pub positive_diffs: bool,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44_100,
            fps: 100.0,
            frame_sizes: [1024, 2048, 4096],
            num_bands: [3, 6, 12],
            fmin: 30.0,
            fmax: 17_000.0,
            diff_ratio: 0.5,
            positive_diffs: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Path to exported model JSON (produced by tools/regen/python/export_downbeats_model.py).
    pub model_json: String,
    /// Path to exported model weights NPZ.
    pub weights_npz: String,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_json: "../models/downbeats_blstm.json".to_string(),
            weights_npz: "../models/downbeats_blstm_weights.npz".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbnConfig {
    pub beats_per_bar: Vec<usize>,
    pub min_bpm: f32,
    pub max_bpm: f32,
    pub num_tempi: usize,
    pub transition_lambda: f32,
    pub observation_lambda: usize,
    pub threshold: f32,
    pub correct: bool,
}

impl Default for DbnConfig {
    fn default() -> Self {
        Self {
            beats_per_bar: vec![3, 4],
            min_bpm: 55.0,
            max_bpm: 215.0,
            num_tempi: 60,
            transition_lambda: 100.0,
            observation_lambda: 16,
            threshold: 0.05,
            correct: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CoreConfig {
    pub feature: FeatureConfig,
    pub model: ModelConfig,
    pub dbn: DbnConfig,
}
