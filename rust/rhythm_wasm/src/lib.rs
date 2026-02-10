use js_sys::Function;
use rhythm_core::{
    analyze, analyze_with_model_data, analyze_with_progress_and_model_data, CoreConfig,
    ProgressEvent, ProgressSink,
};
use serde_wasm_bindgen::to_value;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn analyze_json(
    samples: &[f32],
    sample_rate: u32,
    config_json: Option<String>,
) -> Result<JsValue, JsValue> {
    let config = match config_json {
        Some(json) if !json.trim().is_empty() => serde_json::from_str::<CoreConfig>(&json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?,
        _ => CoreConfig::default(),
    };

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
    let config = match config_json {
        Some(json) if !json.trim().is_empty() => serde_json::from_str::<CoreConfig>(&json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?,
        _ => CoreConfig::default(),
    };

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
    let config = match config_json {
        Some(json) if !json.trim().is_empty() => serde_json::from_str::<CoreConfig>(&json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?,
        _ => CoreConfig::default(),
    };

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
