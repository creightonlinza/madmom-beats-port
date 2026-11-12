// SPDX-License-Identifier: BSD-2-Clause

use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureConfig {
    pub sample_rate: u32,
    pub fps: f32,
    pub frame_sizes: [usize; 3],
    pub num_bands: [usize; 3],
    pub fmin: f32,
    pub fmax: f32,
    pub diff_ratio: f32,
    pub positive_diffs: bool,
    /// The samples are 16-bit PCM as `i16 / 32768`; scale them as madmom
    /// scales a 16-bit signal (by `1 / i16::MAX`) instead of as floats.
    #[serde(default)]
    pub int16_scaling: bool,
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
            int16_scaling: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
pub struct CoreConfig {
    pub feature: FeatureConfig,
    pub model: ModelConfig,
    pub dbn: DbnConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigValidationIssue {
    pub path: String,
    pub message: String,
}

impl ConfigValidationIssue {
    fn new(path: &str, message: impl Into<String>) -> Self {
        Self {
            path: path.to_string(),
            message: message.into(),
        }
    }
}

impl fmt::Display for ConfigValidationIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

pub fn validate_core_config(config: &CoreConfig) -> Result<(), ConfigValidationIssue> {
    let feature = &config.feature;
    if feature.sample_rate == 0 {
        return Err(ConfigValidationIssue::new(
            "feature.sample_rate",
            "must be > 0",
        ));
    }
    if !feature.fps.is_finite() || feature.fps <= 0.0 || feature.fps > feature.sample_rate as f32 {
        return Err(ConfigValidationIssue::new(
            "feature.fps",
            "must be > 0 and <= feature.sample_rate",
        ));
    }
    if feature
        .frame_sizes
        .iter()
        .any(|&n| !(4..=1 << 16).contains(&n) || !n.is_power_of_two())
    {
        return Err(ConfigValidationIssue::new(
            "feature.frame_sizes",
            "all values must be powers of two from 4 to 65536",
        ));
    }
    if feature.frame_sizes.windows(2).any(|w| w[0] >= w[1]) {
        return Err(ConfigValidationIssue::new(
            "feature.frame_sizes",
            "values must be strictly increasing",
        ));
    }
    if feature.num_bands.iter().any(|&n| !(1..=96).contains(&n)) {
        return Err(ConfigValidationIssue::new(
            "feature.num_bands",
            "all values must be from 1 to 96",
        ));
    }
    if !feature.fmin.is_finite() || feature.fmin <= 0.0 {
        return Err(ConfigValidationIssue::new("feature.fmin", "must be > 0"));
    }
    if !feature.fmax.is_finite() || feature.fmax <= feature.fmin {
        return Err(ConfigValidationIssue::new(
            "feature.fmax",
            "must be > feature.fmin",
        ));
    }
    if !feature.diff_ratio.is_finite() || !(0.0..=1.0).contains(&feature.diff_ratio) {
        return Err(ConfigValidationIssue::new(
            "feature.diff_ratio",
            "must be in [0, 1]",
        ));
    }

    let model = &config.model;
    if model.model_json.trim().is_empty() {
        return Err(ConfigValidationIssue::new(
            "model.model_json",
            "must not be empty",
        ));
    }
    if model.weights_npz.trim().is_empty() {
        return Err(ConfigValidationIssue::new(
            "model.weights_npz",
            "must not be empty",
        ));
    }

    let dbn = &config.dbn;
    if dbn.beats_per_bar.is_empty() {
        return Err(ConfigValidationIssue::new(
            "dbn.beats_per_bar",
            "must contain at least one meter",
        ));
    }
    if dbn.beats_per_bar.iter().any(|&n| !(1..=32).contains(&n)) {
        return Err(ConfigValidationIssue::new(
            "dbn.beats_per_bar",
            "all values must be from 1 to 32",
        ));
    }
    if !dbn.min_bpm.is_finite() || dbn.min_bpm <= 0.0 {
        return Err(ConfigValidationIssue::new("dbn.min_bpm", "must be > 0"));
    }
    if !dbn.max_bpm.is_finite() || dbn.max_bpm <= dbn.min_bpm {
        return Err(ConfigValidationIssue::new(
            "dbn.max_bpm",
            "must be > dbn.min_bpm",
        ));
    }
    // The longest beat interval bounds the DBN state count (about its square).
    if 60.0 * feature.fps / dbn.min_bpm > 1000.0 {
        return Err(ConfigValidationIssue::new(
            "dbn.min_bpm",
            "must be >= 0.06 * feature.fps",
        ));
    }
    // The shortest beat interval, rounded as the DBN rounds it, must be at least one frame.
    if (60.0 * feature.fps as f64 / dbn.max_bpm as f64).round_ties_even() < 1.0 {
        return Err(ConfigValidationIssue::new(
            "dbn.max_bpm",
            "must be < 120 * feature.fps",
        ));
    }
    if dbn.num_tempi == 0 {
        return Err(ConfigValidationIssue::new("dbn.num_tempi", "must be > 0"));
    }
    if !dbn.transition_lambda.is_finite() || dbn.transition_lambda <= 0.0 {
        return Err(ConfigValidationIssue::new(
            "dbn.transition_lambda",
            "must be > 0",
        ));
    }
    if dbn.observation_lambda < 2 {
        return Err(ConfigValidationIssue::new(
            "dbn.observation_lambda",
            "must be >= 2",
        ));
    }
    if !dbn.threshold.is_finite() || !(0.0..=1.0).contains(&dbn.threshold) {
        return Err(ConfigValidationIssue::new(
            "dbn.threshold",
            "must be in [0, 1]",
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        let config = CoreConfig::default();
        assert!(validate_core_config(&config).is_ok());
    }

    #[test]
    fn validation_reports_field_path() {
        let mut config = CoreConfig::default();
        config.dbn.num_tempi = 0;

        let issue = validate_core_config(&config).expect_err("expected invalid config");
        assert_eq!(issue.path, "dbn.num_tempi");
    }

    #[test]
    fn config_deserialize_rejects_unknown_fields() {
        let json = r#"
        {
          "feature": {
            "sample_rate": 44100,
            "fps": 100.0,
            "frame_sizes": [1024, 2048, 4096],
            "num_bands": [3, 6, 12],
            "fmin": 30.0,
            "fmax": 17000.0,
            "diff_ratio": 0.5,
            "positive_diffs": true,
            "unknown_feature_field": 1
          },
          "model": {
            "model_json": "../models/downbeats_blstm.json",
            "weights_npz": "../models/downbeats_blstm_weights.npz"
          },
          "dbn": {
            "beats_per_bar": [3, 4],
            "min_bpm": 55.0,
            "max_bpm": 215.0,
            "num_tempi": 60,
            "transition_lambda": 100.0,
            "observation_lambda": 16,
            "threshold": 0.05,
            "correct": true
          }
        }
        "#;
        assert!(serde_json::from_str::<CoreConfig>(json).is_err());
    }
}
