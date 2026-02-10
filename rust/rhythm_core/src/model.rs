use crate::features::Features;
use crate::types::{ActivationOutput, ProgressEvent, ProgressSink, ProgressStage};
use crate::{CoreConfig, RhythmError};
use ndarray::{s, Array1, Array2, Axis};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct ExportedModel {
    version: u32,
    networks: Vec<ExportedNetwork>,
}

#[derive(Debug, Deserialize)]
struct ExportedNetwork {
    layers: Vec<ExportedLayer>,
}

#[derive(Debug, Deserialize)]
#[allow(clippy::large_enum_variant)]
#[serde(tag = "type")]
enum ExportedLayer {
    #[serde(rename = "feedforward")]
    FeedForward {
        weights: String,
        bias: String,
        activation: String,
    },
    #[serde(rename = "recurrent")]
    Recurrent {
        weights: String,
        bias: String,
        recurrent_weights: String,
        activation: String,
    },
    #[serde(rename = "lstm")]
    Lstm {
        input_gate: ExportedGate,
        forget_gate: ExportedGate,
        cell: ExportedGate,
        output_gate: ExportedGate,
        activation: String,
    },
    #[serde(rename = "bidirectional")]
    Bidirectional {
        fwd: Box<ExportedLayer>,
        bwd: Box<ExportedLayer>,
    },
}

#[derive(Debug, Deserialize)]
struct ExportedGate {
    weights: String,
    bias: String,
    recurrent_weights: String,
    peephole_weights: Option<String>,
    activation: String,
}

#[derive(Debug, Clone, Copy)]
enum Activation {
    Linear,
    Tanh,
    Sigmoid,
    Relu,
    Softmax,
    Elu,
}

impl Activation {
    fn from_name(name: &str) -> Result<Self, RhythmError> {
        match name {
            "linear" => Ok(Self::Linear),
            "tanh" => Ok(Self::Tanh),
            "sigmoid" => Ok(Self::Sigmoid),
            "relu" => Ok(Self::Relu),
            "softmax" => Ok(Self::Softmax),
            "elu" => Ok(Self::Elu),
            _ => Err(RhythmError::Model(format!(
                "unsupported activation: {}",
                name
            ))),
        }
    }
}

pub fn run_inference(
    features: &Features,
    config: &CoreConfig,
    progress: &mut dyn ProgressSink,
) -> Result<ActivationOutput, RhythmError> {
    let log_progress = std::env::var("RHYTHM_PROGRESS").ok().as_deref() == Some("1");
    let model = load_model(&config.model.model_json, &config.model.weights_npz)?;
    run_inference_with_model(features, &model, log_progress, progress)
}

pub fn run_inference_with_data(
    features: &Features,
    model_json: &str,
    weights_npz: &[u8],
    progress: &mut dyn ProgressSink,
) -> Result<ActivationOutput, RhythmError> {
    let log_progress = std::env::var("RHYTHM_PROGRESS").ok().as_deref() == Some("1");
    let model = load_model_from_data(model_json, weights_npz)?;
    run_inference_with_model(features, &model, log_progress, progress)
}

fn run_inference_with_model(
    features: &Features,
    model: &LoadedModel,
    log_progress: bool,
    sink: &mut dyn ProgressSink,
) -> Result<ActivationOutput, RhythmError> {
    let mut predictions: Option<Array2<f32>> = None;
    let total = model.meta.networks.len().max(1) as f32;
    for (idx, network) in model.meta.networks.iter().enumerate() {
        if log_progress {
            println!(
                "inference: network {} of {} (frames {})",
                idx + 1,
                model.meta.networks.len(),
                features.data.len_of(Axis(0))
            );
        }
        let out = run_network(network, model, &features.data, log_progress)?;
        predictions = Some(match predictions {
            None => out,
            Some(prev) => &prev + &out,
        });
        sink.on_progress(ProgressEvent {
            stage: ProgressStage::Inference,
            progress: (idx as f32 + 1.0) / total,
        });
    }

    let mut averaged = predictions
        .ok_or_else(|| RhythmError::Model("no networks in exported model".to_string()))?;
    let count = model.meta.networks.len() as f32;
    averaged.mapv_inplace(|v| v / count);

    let output = if averaged.len_of(Axis(1)) == 3 {
        // drop non-beat column (index 0)
        averaged.slice(s![.., 1..3]).to_owned()
    } else {
        averaged
    };

    if output.len_of(Axis(1)) < 2 {
        return Err(RhythmError::Model(
            "model output must have at least 2 columns".to_string(),
        ));
    }

    let beat = output.column(0).to_vec();
    let downbeat = output.column(1).to_vec();
    Ok(ActivationOutput { beat, downbeat })
}

struct LoadedModel {
    meta: ExportedModel,
    arrays: crate::io::NpzArrays,
}

fn load_model(model_json: &str, weights_npz: &str) -> Result<LoadedModel, RhythmError> {
    let json_text =
        std::fs::read_to_string(model_json).map_err(|e| RhythmError::Io(e.to_string()))?;
    let meta: ExportedModel =
        serde_json::from_str(&json_text).map_err(|e| RhythmError::Model(e.to_string()))?;
    if meta.version != 1 {
        return Err(RhythmError::Model(format!(
            "unsupported model version {}",
            meta.version
        )));
    }
    let arrays = crate::io::NpzArrays::open(weights_npz)?;
    Ok(LoadedModel { meta, arrays })
}

fn load_model_from_data(model_json: &str, weights_npz: &[u8]) -> Result<LoadedModel, RhythmError> {
    let meta: ExportedModel =
        serde_json::from_str(model_json).map_err(|e| RhythmError::Model(e.to_string()))?;
    if meta.version != 1 {
        return Err(RhythmError::Model(format!(
            "unsupported model version {}",
            meta.version
        )));
    }
    let arrays = crate::io::NpzArrays::from_bytes(weights_npz)?;
    Ok(LoadedModel { meta, arrays })
}

fn run_network(
    network: &ExportedNetwork,
    model: &LoadedModel,
    input: &Array2<f32>,
    progress: bool,
) -> Result<Array2<f32>, RhythmError> {
    let mut data = input.clone();
    for layer in &network.layers {
        data = apply_layer(layer, model, &data, progress)?;
    }
    Ok(data)
}

fn apply_layer(
    layer: &ExportedLayer,
    model: &LoadedModel,
    data: &Array2<f32>,
    progress: bool,
) -> Result<Array2<f32>, RhythmError> {
    match layer {
        ExportedLayer::FeedForward {
            weights,
            bias,
            activation,
        } => {
            let w = model.arrays.array2(weights)?;
            let b = model.arrays.array1(bias)?;
            let mut out = data.dot(&w);
            add_bias(&mut out, &b);
            apply_activation(&mut out, Activation::from_name(activation)?)?;
            Ok(out)
        }
        ExportedLayer::Recurrent {
            weights,
            bias,
            recurrent_weights,
            activation,
        } => {
            let w = model.arrays.array2(weights)?;
            let b = model.arrays.array1(bias)?;
            let r = model.arrays.array2(recurrent_weights)?;
            let act = Activation::from_name(activation)?;
            let mut out = data.dot(&w);
            add_bias(&mut out, &b);
            let mut prev = Array1::<f32>::zeros(b.len());
            let step = progress_step(out.len_of(Axis(0)));
            for i in 0..out.len_of(Axis(0)) {
                if progress && step > 0 && i % step == 0 {
                    print_progress("recurrent", i, out.len_of(Axis(0)));
                }
                let mut row = out.row(i).to_owned();
                row += &r.t().dot(&prev);
                apply_activation_row(&mut row, act)?;
                prev = row.clone();
                out.row_mut(i).assign(&row);
            }
            Ok(out)
        }
        ExportedLayer::Lstm {
            input_gate,
            forget_gate,
            cell,
            output_gate,
            activation,
        } => {
            let act = Activation::from_name(activation)?;
            lstm_layer(
                data,
                model,
                input_gate,
                forget_gate,
                cell,
                output_gate,
                act,
                progress,
            )
        }
        ExportedLayer::Bidirectional { fwd, bwd } => {
            let fwd_out = apply_layer(fwd, model, data, progress)?;
            let mut reversed = data.clone();
            reversed.invert_axis(Axis(0));
            let mut bwd_out = apply_layer(bwd, model, &reversed, progress)?;
            bwd_out.invert_axis(Axis(0));
            hstack(&fwd_out, &bwd_out)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn lstm_layer(
    data: &Array2<f32>,
    model: &LoadedModel,
    input_gate: &ExportedGate,
    forget_gate: &ExportedGate,
    cell: &ExportedGate,
    output_gate: &ExportedGate,
    activation: Activation,
    progress: bool,
) -> Result<Array2<f32>, RhythmError> {
    let ig = Gate::load(model, input_gate)?;
    let fg = Gate::load(model, forget_gate)?;
    let cg = Gate::load(model, cell)?;
    let og = Gate::load(model, output_gate)?;

    let size = data.len_of(Axis(0));
    let hidden = ig.bias.len();
    let mut out = Array2::<f32>::zeros((size, hidden));

    let mut prev = Array1::<f32>::zeros(hidden);
    let mut state = Array1::<f32>::zeros(hidden);

    let step = progress_step(size);
    for i in 0..size {
        if progress && step > 0 && i % step == 0 {
            print_progress("lstm", i, size);
        }
        let x = data.row(i).to_owned();
        let ig_out = ig.activate(&x, &prev, Some(&state))?;
        let fg_out = fg.activate(&x, &prev, Some(&state))?;
        let cell_out = cg.activate(&x, &prev, None)?;
        state = &cell_out * &ig_out + &state * &fg_out;
        let og_out = og.activate(&x, &prev, Some(&state))?;
        let mut activated = state.clone();
        apply_activation_row(&mut activated, activation)?;
        let row = &activated * &og_out;
        prev = row.clone();
        out.row_mut(i).assign(&row);
    }
    Ok(out)
}

struct Gate {
    weights: Array2<f32>,
    bias: Array1<f32>,
    recurrent: Array2<f32>,
    peephole: Option<Array1<f32>>,
    activation: Activation,
}

impl Gate {
    fn load(model: &LoadedModel, gate: &ExportedGate) -> Result<Self, RhythmError> {
        Ok(Self {
            weights: model.arrays.array2(&gate.weights)?,
            bias: model.arrays.array1(&gate.bias)?,
            recurrent: model.arrays.array2(&gate.recurrent_weights)?,
            peephole: match &gate.peephole_weights {
                Some(name) => Some(model.arrays.array1(name)?),
                None => None,
            },
            activation: Activation::from_name(&gate.activation)?,
        })
    }

    fn activate(
        &self,
        input: &Array1<f32>,
        prev: &Array1<f32>,
        state: Option<&Array1<f32>>,
    ) -> Result<Array1<f32>, RhythmError> {
        let mut out = self.weights.t().dot(input) + &self.bias;
        if let Some(ph) = &self.peephole {
            if let Some(st) = state {
                out += &(st * ph);
            }
        }
        out += &self.recurrent.t().dot(prev);
        apply_activation_row(&mut out, self.activation)?;
        Ok(out)
    }
}

fn add_bias(out: &mut Array2<f32>, bias: &Array1<f32>) {
    for mut row in out.rows_mut() {
        row += bias;
    }
}

fn apply_activation(out: &mut Array2<f32>, act: Activation) -> Result<(), RhythmError> {
    for i in 0..out.len_of(Axis(0)) {
        let mut row = out.row(i).to_owned();
        apply_activation_row(&mut row, act)?;
        out.row_mut(i).assign(&row);
    }
    Ok(())
}

fn apply_activation_row(row: &mut Array1<f32>, act: Activation) -> Result<(), RhythmError> {
    match act {
        Activation::Linear => {}
        Activation::Tanh => row.mapv_inplace(|v| v.tanh()),
        Activation::Sigmoid => row.mapv_inplace(|v| 0.5 * (1.0 + (0.5 * v).tanh())),
        Activation::Relu => row.mapv_inplace(|v| if v > 0.0 { v } else { 0.0 }),
        Activation::Elu => row.mapv_inplace(|v| if v > 0.0 { v } else { v.exp() - 1.0 }),
        Activation::Softmax => {
            let max = row.iter().cloned().fold(f32::MIN, f32::max);
            let mut sum = 0.0;
            for v in row.iter_mut() {
                *v = (*v - max).exp();
                sum += *v;
            }
            if sum > 0.0 {
                for v in row.iter_mut() {
                    *v /= sum;
                }
            }
        }
    }
    Ok(())
}

fn hstack(a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>, RhythmError> {
    if a.len_of(Axis(0)) != b.len_of(Axis(0)) {
        return Err(RhythmError::Model(
            "frame count mismatch in bidirectional hstack".to_string(),
        ));
    }
    let rows = a.len_of(Axis(0));
    let cols = a.len_of(Axis(1)) + b.len_of(Axis(1));
    let mut out = Array2::<f32>::zeros((rows, cols));
    out.slice_mut(s![.., 0..a.len_of(Axis(1))]).assign(a);
    out.slice_mut(s![.., a.len_of(Axis(1))..]).assign(b);
    Ok(out)
}

fn progress_step(size: usize) -> usize {
    if size >= 2000 {
        (size / 10).max(1)
    } else {
        0
    }
}

fn print_progress(label: &str, current: usize, total: usize) {
    let pct = if total == 0 {
        0.0
    } else {
        (current as f32 / total as f32) * 100.0
    };
    println!("{} progress: {:.0}% ({}/{})", label, pct, current, total);
}
