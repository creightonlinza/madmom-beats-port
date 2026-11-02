// SPDX-License-Identifier: BSD-2-Clause

use js_sys::Function;
use madmom_beats_port_core::{
    analyze, analyze_with_model_data, analyze_with_progress_and_model_data, decode_network_outputs,
    extract_features, feature_columns, validate_core_config, AnalysisOutput, BeatModel, CoreConfig,
    ProgressEvent, ProgressSink,
};
use ndarray::Array2;
use serde::Serialize;
use serde_wasm_bindgen::to_value;
use wasm_bindgen::prelude::*;

#[derive(Serialize)]
struct WasmConfigError {
    code: &'static str,
    message: String,
    path: Option<String>,
    context: Option<String>,
}

fn parse_config(config_json: Option<String>) -> Result<CoreConfig, WasmConfigError> {
    let config = match config_json {
        Some(json) if !json.trim().is_empty() => serde_json::from_str::<CoreConfig>(&json)
            .map_err(|err| WasmConfigError {
                code: "CONFIG_PARSE_ERROR",
                message: format!("failed to parse config_json: {err}"),
                path: Some("config_json".to_string()),
                context: if err.line() > 0 {
                    Some(format!("line {}, column {}", err.line(), err.column()))
                } else {
                    None
                },
            })?,
        _ => CoreConfig::default(),
    };

    validate_core_config(&config).map_err(|issue| WasmConfigError {
        code: "CONFIG_VALIDATION_ERROR",
        message: format!("invalid config: {}", issue.message),
        path: Some(issue.path),
        context: None,
    })?;

    Ok(config)
}

fn as_js_error(err: WasmConfigError) -> JsValue {
    serde_wasm_bindgen::to_value(&err).unwrap_or_else(|_| JsValue::from_str(&err.message))
}

#[wasm_bindgen]
pub fn default_config_json() -> Result<String, JsValue> {
    serde_json::to_string_pretty(&CoreConfig::default())
        .map_err(|err| JsValue::from_str(&err.to_string()))
}

#[wasm_bindgen]
pub fn validate_config_json(config_json: Option<String>) -> JsValue {
    match parse_config(config_json) {
        Ok(_) => JsValue::NULL,
        Err(err) => as_js_error(err),
    }
}

#[wasm_bindgen]
pub fn analyze_json(
    samples: &[f32],
    sample_rate: u32,
    config_json: Option<String>,
) -> Result<JsValue, JsValue> {
    let config = parse_config(config_json).map_err(as_js_error)?;

    let output =
        analyze(samples, sample_rate, &config).map_err(|e| JsValue::from_str(&e.to_string()))?;
    to_value(&output).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub fn analyze_json_with_model(
    samples: &[f32],
    sample_rate: u32,
    config_json: Option<String>,
    model_json: String,
    weights_npz: Vec<u8>,
) -> Result<JsValue, JsValue> {
    let config = parse_config(config_json).map_err(as_js_error)?;

    let output = analyze_with_model_data(samples, sample_rate, &config, &model_json, &weights_npz)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    to_value(&output).map_err(|e| JsValue::from_str(&e.to_string()))
}

struct JsProgressSink {
    cb: Function,
}

impl ProgressSink for JsProgressSink {
    fn on_progress(&mut self, event: ProgressEvent) {
        let _ = self.cb.call2(
            &JsValue::NULL,
            &JsValue::from(event.stage as u32),
            &JsValue::from_f64(event.progress as f64),
        );
    }
}

#[wasm_bindgen]
pub fn analyze_json_with_model_progress(
    samples: &[f32],
    sample_rate: u32,
    config_json: Option<String>,
    model_json: String,
    weights_npz: Vec<u8>,
    progress_cb: Function,
) -> Result<JsValue, JsValue> {
    let config = parse_config(config_json).map_err(as_js_error)?;

    let mut sink = JsProgressSink { cb: progress_cb };
    let output = analyze_with_progress_and_model_data(
        samples,
        sample_rate,
        &config,
        Some(&mut sink),
        &model_json,
        &weights_npz,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    to_value(&output).map_err(|e| JsValue::from_str(&e.to_string()))
}

// Staged analysis for running the networks and bar lengths in parallel
// (one Web Worker each): `extract_features`, then `BeatNetworks.run` for every
// network, then `decode_beats` on the outputs in network order, optionally
// once per bar length. Same result as `analyze_json_with_model`.

fn js_error(e: impl ToString) -> JsValue {
    JsValue::from_str(&e.to_string())
}

/// Number of feature columns `extract_features_json` returns per frame.
#[wasm_bindgen]
pub fn feature_columns_json(config_json: Option<String>) -> Result<usize, JsValue> {
    let config = parse_config(config_json).map_err(as_js_error)?;
    feature_columns(&config).map_err(js_error)
}

/// Features, frames x `feature_columns_json` in row order.
#[wasm_bindgen]
pub fn extract_features_json(
    samples: &[f32],
    sample_rate: u32,
    config_json: Option<String>,
) -> Result<Vec<f32>, JsValue> {
    let config = parse_config(config_json).map_err(as_js_error)?;
    let features = extract_features(samples, sample_rate, &config).map_err(js_error)?;
    Ok(features.into_raw_vec())
}

#[wasm_bindgen]
pub struct BeatNetworks {
    model: BeatModel,
    features: Option<Array2<f32>>,
}

#[wasm_bindgen]
impl BeatNetworks {
    #[wasm_bindgen(constructor)]
    pub fn new(model_json: String, weights_npz: Vec<u8>) -> Result<BeatNetworks, JsValue> {
        Ok(BeatNetworks {
            model: BeatModel::from_data(&model_json, &weights_npz).map_err(js_error)?,
            features: None,
        })
    }

    pub fn count(&self) -> usize {
        self.model.num_networks()
    }

    /// Keeps the features (from `extract_features_json`) for `run`.
    pub fn set_features(&mut self, features: Vec<f32>, columns: usize) -> Result<(), JsValue> {
        if columns == 0 || !features.len().is_multiple_of(columns) {
            return Err(js_error("features do not divide into the column count"));
        }
        let frames = features.len() / columns;
        self.features =
            Some(Array2::from_shape_vec((frames, columns), features).map_err(js_error)?);
        Ok(())
    }

    /// Network `index`'s output, frames x outputs in row order.
    pub fn run(&self, index: usize) -> Result<Vec<f32>, JsValue> {
        let features = self
            .features
            .as_ref()
            .ok_or_else(|| js_error("set_features first"))?;
        let out = self.model.run_network(features, index).map_err(js_error)?;
        Ok(out.as_standard_layout().into_owned().into_raw_vec())
    }
}

#[derive(Serialize)]
struct DecodedBeats {
    /// The winning bar model's log probability; absent where no bar length fits.
    log_probability: Option<f64>,
    output: Option<AnalysisOutput>,
}

/// Splits outputs concatenated in network order into one array per network.
fn split_network_outputs(
    outputs: &[f32],
    networks: usize,
    columns: usize,
) -> Result<Vec<Array2<f32>>, String> {
    let values_per_frame = networks.saturating_mul(columns);
    if values_per_frame == 0
        || outputs.is_empty()
        || !outputs.len().is_multiple_of(values_per_frame)
    {
        return Err("network outputs do not match the network and column counts".to_string());
    }
    let frames = outputs.len() / values_per_frame;
    outputs
        .chunks_exact(frames * columns)
        .map(|chunk| {
            Array2::from_shape_vec((frames, columns), chunk.to_vec()).map_err(|e| e.to_string())
        })
        .collect()
}

/// Decodes beats from the networks' outputs, concatenated in network order
/// (each frames x `columns`).
#[wasm_bindgen]
pub fn decode_beats(
    outputs: &[f32],
    networks: usize,
    columns: usize,
    config_json: Option<String>,
    progress_cb: Option<Function>,
) -> Result<JsValue, JsValue> {
    let config = parse_config(config_json).map_err(as_js_error)?;
    let arrays = split_network_outputs(outputs, networks, columns).map_err(js_error)?;
    let mut sink = progress_cb.map(|cb| JsProgressSink { cb });
    let decoded = decode_network_outputs(
        arrays,
        &config,
        sink.as_mut().map(|s| s as &mut dyn ProgressSink),
    )
    .map_err(js_error)?;
    let (output, log_probability) = match decoded {
        Some((output, logp)) => (Some(output), Some(logp)),
        None => (None, None),
    };
    to_value(&DecodedBeats {
        log_probability,
        output,
    })
    .map_err(js_error)
}

#[cfg(test)]
mod tests {
    use super::split_network_outputs;
    use madmom_beats_port_core::{analyze_with_model_data, CoreConfig};

    const MODEL_JSON: &str = include_str!("../../../models/downbeats_blstm.json");
    const MODEL_WEIGHTS: &[u8] = include_bytes!("../../../models/downbeats_blstm_weights.npz");

    #[test]
    fn split_network_outputs_rejects_inputs_that_do_not_divide() {
        assert!(split_network_outputs(&[], 8, 3).is_err());
        assert!(split_network_outputs(&[0.0; 24], 0, 3).is_err());
        assert!(split_network_outputs(&[0.0; 24], 8, 0).is_err());
        assert!(split_network_outputs(&[0.0; 7], 8, 3).is_err());
    }

    #[test]
    fn split_network_outputs_keeps_network_order() {
        let outputs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let arrays = split_network_outputs(&outputs, 2, 2).unwrap();
        assert_eq!(arrays.len(), 2);
        assert_eq!(arrays[0].dim(), (2, 2));
        assert_eq!(arrays[0][(1, 0)], 3.0);
        assert_eq!(arrays[1][(0, 0)], 5.0);
    }

    #[test]
    #[ignore = "long-running stability test"]
    fn long_synthetic_track_does_not_trap() {
        let sample_rate = 44_100u32;
        let duration_sec = 11 * 60;
        let total_samples = sample_rate as usize * duration_sec;

        let mut samples = vec![0.0f32; total_samples];
        let beat_period = (sample_rate as usize * 60) / 120;
        for idx in (0..total_samples).step_by(beat_period.max(1)) {
            for tap in 0..256usize {
                let pos = idx + tap;
                if pos >= total_samples {
                    break;
                }
                samples[pos] += (1.0 - tap as f32 / 256.0) * 0.5;
            }
        }

        let config = CoreConfig::default();
        let result =
            analyze_with_model_data(&samples, sample_rate, &config, MODEL_JSON, MODEL_WEIGHTS);

        assert!(result.is_ok(), "long synthetic model-data analysis failed");
    }
}
