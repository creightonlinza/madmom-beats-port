use js_sys::Function;
use madmom_beats_port_core::{
    analyze, analyze_with_model_data, analyze_with_progress_and_model_data, validate_core_config,
    CoreConfig, ProgressEvent, ProgressSink,
};
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

#[cfg(test)]
mod tests {
    use madmom_beats_port_core::{analyze_with_model_data, CoreConfig};

    const MODEL_JSON: &str = include_str!("../../../models/downbeats_blstm.json");
    const MODEL_WEIGHTS: &[u8] = include_bytes!("../../../models/downbeats_blstm_weights.npz");

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
