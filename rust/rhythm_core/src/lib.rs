//! Cross-platform beat + downbeat detection core.

mod config;
mod dbn;
mod features;
mod io;
mod model;
mod types;

pub use config::{CoreConfig, DbnConfig, FeatureConfig, ModelConfig};
pub use types::{
    ActivationOutput, AnalysisOutput, BeatEvent, DownbeatEvent, ProgressEvent, ProgressSink,
    ProgressStage,
};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum RhythmError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("model error: {0}")]
    Model(String),
    #[error("not implemented: {0}")]
    NotImplemented(String),
    #[error("io error: {0}")]
    Io(String),
}

/// Analyze a full track of mono PCM samples at 44.1kHz.
///
/// Returns per-frame activations and decoded beat/downbeat events.
pub fn analyze(
    samples: &[f32],
    sample_rate: u32,
    config: &CoreConfig,
) -> Result<AnalysisOutput, RhythmError> {
    analyze_with_progress(samples, sample_rate, config, None)
}

/// Analyze a full track using in-memory model data (for WASM).
pub fn analyze_with_model_data(
    samples: &[f32],
    sample_rate: u32,
    config: &CoreConfig,
    model_json: &str,
    weights_npz: &[u8],
) -> Result<AnalysisOutput, RhythmError> {
    analyze_with_progress_and_model_data(
        samples,
        sample_rate,
        config,
        None,
        model_json,
        weights_npz,
    )
}

/// Analyze with optional progress callbacks (0.0..=1.0 for each stage).
pub fn analyze_with_progress(
    samples: &[f32],
    sample_rate: u32,
    config: &CoreConfig,
    progress: Option<&mut dyn ProgressSink>,
) -> Result<AnalysisOutput, RhythmError> {
    if sample_rate != config.feature.sample_rate {
        return Err(RhythmError::InvalidInput(format!(
            "expected {} Hz mono samples; got {}",
            config.feature.sample_rate, sample_rate
        )));
    }

    let mut noop = NoopProgressSink;
    let sink: &mut dyn ProgressSink = match progress {
        Some(s) => s,
        None => &mut noop,
    };

    let features = features::compute_features(samples, config, sink)?;
    let activations = model::run_inference(&features, config, sink)?;
    let events = dbn::decode(&activations, config, sink)?;

    Ok(AnalysisOutput {
        activations,
        events,
    })
}

/// Analyze using in-memory model data and optional progress callbacks.
pub fn analyze_with_progress_and_model_data(
    samples: &[f32],
    sample_rate: u32,
    config: &CoreConfig,
    progress: Option<&mut dyn ProgressSink>,
    model_json: &str,
    weights_npz: &[u8],
) -> Result<AnalysisOutput, RhythmError> {
    if sample_rate != config.feature.sample_rate {
        return Err(RhythmError::InvalidInput(format!(
            "expected {} Hz mono samples; got {}",
            config.feature.sample_rate, sample_rate
        )));
    }

    let mut noop = NoopProgressSink;
    let sink: &mut dyn ProgressSink = match progress {
        Some(s) => s,
        None => &mut noop,
    };

    let features = features::compute_features(samples, config, sink)?;
    let activations = model::run_inference_with_data(&features, model_json, weights_npz, sink)?;
    let events = dbn::decode(&activations, config, sink)?;

    Ok(AnalysisOutput {
        activations,
        events,
    })
}

struct NoopProgressSink;

impl ProgressSink for NoopProgressSink {
    fn on_progress(&mut self, _event: ProgressEvent) {}
}
