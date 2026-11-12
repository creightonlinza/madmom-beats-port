// SPDX-License-Identifier: BSD-2-Clause
// Ported from madmom (BSD 2-Clause, Copyright (c) 2012-2014 JKU / OFAI). See NOTICE.

use crate::features::Features;
use crate::numeric;
use crate::types::{ActivationOutput, ProgressEvent, ProgressSink, ProgressStage};
use crate::{CoreConfig, RhythmError};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, Axis};
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
    let log_progress = std::env::var("MADMOM_BEATS_PORT_PROGRESS").ok().as_deref() == Some("1");
    let model = load_model(&config.model.model_json, &config.model.weights_npz)?;
    run_inference_with_model(features, &model, log_progress, progress)
}

pub fn run_inference_with_data(
    features: &Features,
    model_json: &str,
    weights_npz: &[u8],
    progress: &mut dyn ProgressSink,
) -> Result<ActivationOutput, RhythmError> {
    let log_progress = std::env::var("MADMOM_BEATS_PORT_PROGRESS").ok().as_deref() == Some("1");
    let model = load_model_from_data(model_json, weights_npz)?;
    run_inference_with_model(features, &model, log_progress, progress)
}

fn run_inference_with_model(
    features: &Features,
    model: &LoadedModel,
    log_progress: bool,
    sink: &mut dyn ProgressSink,
) -> Result<ActivationOutput, RhythmError> {
    let mut sum = None;
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
        sum = Some(add_network_output(sum, out)?);
        sink.on_progress(ProgressEvent {
            stage: ProgressStage::Inference,
            progress: (idx as f32 + 1.0) / total,
        });
    }
    activations_from_sum(sum, model.meta.networks.len())
}

/// The networks' outputs (in network order) averaged into the activations
/// the DBN decodes.
pub(crate) fn average_network_outputs(
    outputs: Vec<Array2<f32>>,
) -> Result<ActivationOutput, RhythmError> {
    let count = outputs.len();
    let mut sum = None;
    for out in outputs {
        sum = Some(add_network_output(sum, out)?);
    }
    activations_from_sum(sum, count)
}

fn add_network_output(
    sum: Option<Array2<f32>>,
    out: Array2<f32>,
) -> Result<Array2<f32>, RhythmError> {
    let Some(mut sum) = sum else {
        return Ok(out);
    };
    if sum.dim() != out.dim() {
        return Err(RhythmError::Model(
            "network output shape mismatch".to_string(),
        ));
    }
    sum += &out;
    Ok(sum)
}

fn activations_from_sum(
    sum: Option<Array2<f32>>,
    count: usize,
) -> Result<ActivationOutput, RhythmError> {
    let mut averaged =
        sum.ok_or_else(|| RhythmError::Model("no networks in exported model".to_string()))?;
    let count = count as f32;
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

/// The beat model's networks, loaded once and run one at a time, so callers
/// can spread them over threads or workers. `average_network_outputs` then
/// combines their outputs exactly as `run_inference` does.
pub struct BeatModel {
    model: LoadedModel,
}

impl BeatModel {
    pub fn from_data(model_json: &str, weights_npz: &[u8]) -> Result<Self, RhythmError> {
        Ok(Self {
            model: load_model_from_data(model_json, weights_npz)?,
        })
    }

    pub fn num_networks(&self) -> usize {
        self.model.meta.networks.len()
    }

    /// Runs network `index` over features shaped frames x feature columns.
    pub fn run_network(
        &self,
        features: &Array2<f32>,
        index: usize,
    ) -> Result<Array2<f32>, RhythmError> {
        let network = self
            .model
            .meta
            .networks
            .get(index)
            .ok_or_else(|| RhythmError::InvalidInput(format!("no network {}", index)))?;
        run_network(network, &self.model, features, false)
    }
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
    let mut data: Option<Array2<f32>> = None;
    for layer in &network.layers {
        let layer_in = data.as_ref().unwrap_or(input);
        data = Some(apply_layer(layer, model, layer_in, progress)?);
    }
    Ok(data.unwrap_or_else(|| input.to_owned()))
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
            let w = model.arrays.array2_view(weights)?;
            let b = model.arrays.array1_view(bias)?;
            let mut out = matmul_plus_bias(data, &w, b)?;
            apply_activation(&mut out, Activation::from_name(activation)?)?;
            Ok(out)
        }
        ExportedLayer::Recurrent {
            weights,
            bias,
            recurrent_weights,
            activation,
        } => {
            let w = model.arrays.array2_view(weights)?;
            let b = model.arrays.array1_view(bias)?;
            let r = model.arrays.array2_view(recurrent_weights)?;
            let act = Activation::from_name(activation)?;
            let mut out = matmul_plus_bias(data, &w, b)?;
            let hidden = out.len_of(Axis(1));
            if r.dim() != (hidden, hidden) {
                return Err(RhythmError::Model(
                    "recurrent weight shape mismatch".to_string(),
                ));
            }
            let mut prev = Array1::<f32>::zeros(hidden);
            let mut recurrent = Array1::<f32>::zeros(hidden);
            let step = progress_step(out.len_of(Axis(0)));
            for i in 0..out.len_of(Axis(0)) {
                if progress && step > 0 && i % step == 0 {
                    print_progress("recurrent", i, out.len_of(Axis(0)));
                }
                vec_matmul(&prev, &r, &mut recurrent);
                let mut row = out.row_mut(i);
                row += &recurrent;
                apply_activation_row(&mut row, act)?;
                prev.assign(&row);
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
    let gates = [
        Gate::load(model, input_gate)?,
        Gate::load(model, forget_gate)?,
        Gate::load(model, cell)?,
        Gate::load(model, output_gate)?,
    ];
    lstm_forward(data, &gates, activation, progress)
}

/// Gates in order: input, forget, cell, output.
fn lstm_forward(
    data: &Array2<f32>,
    gates: &[Gate; 4],
    activation: Activation,
    progress: bool,
) -> Result<Array2<f32>, RhythmError> {
    let ig = &gates[0];

    let size = data.len_of(Axis(0));
    let inputs = data.len_of(Axis(1));
    let hidden = ig.bias.len();
    for gate in gates {
        if hidden == 0
            || gate.bias.len() != hidden
            || gate.weights.dim() != (inputs, hidden)
            || gate.recurrent.dim() != (hidden, hidden)
            || gate.peephole.as_ref().is_some_and(|p| p.len() != hidden)
        {
            return Err(RhythmError::Model("lstm gate shape mismatch".to_string()));
        }
    }

    // All four gates' weights side by side, one row per input: a frame's
    // input (and recurrent) products then accumulate for every unit in one
    // pass over contiguous memory, with many independent sums in flight
    // instead of one long dependency chain. Each unit sums its products from
    // input 0 upwards, starting at zero: the order of madmom's `np.dot` (see
    // `numeric::dot`).
    let width = 4 * hidden;
    let mut input_weights = vec![0.0f32; inputs * width];
    let mut recurrent_weights = vec![0.0f32; hidden * width];
    for (g, gate) in gates.iter().enumerate() {
        for i in 0..inputs {
            for j in 0..hidden {
                input_weights[i * width + g * hidden + j] = gate.weights[(i, j)];
            }
        }
        for k in 0..hidden {
            for j in 0..hidden {
                recurrent_weights[k * width + g * hidden + j] = gate.recurrent[(k, j)];
            }
        }
    }

    let mut out = Array2::<f32>::zeros((size, hidden));
    let mut prev = vec![0.0f32; hidden];
    let mut state = vec![0.0f32; hidden];
    let mut input_sums = vec![0.0f32; width];
    let mut recurrent_sums = vec![0.0f32; width];
    let mut gate_out = vec![0.0f32; width];
    let mut activated = vec![0.0f32; hidden];

    let step = progress_step(size);
    for i in 0..size {
        if progress && step > 0 && i % step == 0 {
            print_progress("lstm", i, size);
        }
        let mut out_row = out.row_mut(i);
        lstm_frame(
            gates,
            activation,
            data.row(i),
            &input_weights,
            &recurrent_weights,
            &mut LstmBuffers {
                prev: &mut prev,
                state: &mut state,
                input_sums: &mut input_sums,
                recurrent_sums: &mut recurrent_sums,
                gate_out: &mut gate_out,
                activated: &mut activated,
            },
            out_row
                .as_slice_mut()
                .expect("lstm output rows are contiguous"),
        )?;
    }
    Ok(out)
}

struct LstmBuffers<'a> {
    prev: &'a mut [f32],
    state: &'a mut [f32],
    input_sums: &'a mut [f32],
    recurrent_sums: &'a mut [f32],
    gate_out: &'a mut [f32],
    activated: &'a mut [f32],
}

/// One LSTM frame. Kept out of line so WebAssembly engines see a function
/// called once per frame and switch it to optimized code early; a network run
/// alone in a worker would otherwise stay in V8's baseline tier.
#[inline(never)]
fn lstm_frame(
    gates: &[Gate; 4],
    activation: Activation,
    x: ArrayView1<'_, f32>,
    input_weights: &[f32],
    recurrent_weights: &[f32],
    buffers: &mut LstmBuffers<'_>,
    out_row: &mut [f32],
) -> Result<(), RhythmError> {
    let [ig, fg, cg, og] = gates;
    let hidden = buffers.prev.len();
    let width = buffers.input_sums.len();
    buffers.input_sums.fill(0.0);
    for (&x, weights) in x.iter().zip(input_weights.chunks_exact(width)) {
        for (sum, &w) in buffers.input_sums.iter_mut().zip(weights) {
            *sum += x * w;
        }
    }
    buffers.recurrent_sums.fill(0.0);
    for (&p, weights) in buffers
        .prev
        .iter()
        .zip(recurrent_weights.chunks_exact(width))
    {
        for (sum, &w) in buffers.recurrent_sums.iter_mut().zip(weights) {
            *sum += p * w;
        }
    }
    let (input_sums, recurrent_sums, gate_out) = (
        &*buffers.input_sums,
        &*buffers.recurrent_sums,
        &mut *buffers.gate_out,
    );
    // Input, forget and cell gates see the previous cell state; the cell
    // gate has no peephole.
    ig.activate_from_sums(
        &input_sums[..hidden],
        &recurrent_sums[..hidden],
        Some(buffers.state),
        &mut gate_out[..hidden],
    )?;
    fg.activate_from_sums(
        &input_sums[hidden..2 * hidden],
        &recurrent_sums[hidden..2 * hidden],
        Some(buffers.state),
        &mut gate_out[hidden..2 * hidden],
    )?;
    cg.activate_from_sums(
        &input_sums[2 * hidden..3 * hidden],
        &recurrent_sums[2 * hidden..3 * hidden],
        None,
        &mut gate_out[2 * hidden..3 * hidden],
    )?;
    for j in 0..hidden {
        buffers.state[j] =
            gate_out[2 * hidden + j] * gate_out[j] + buffers.state[j] * gate_out[hidden + j];
    }
    // The output gate sees the updated cell state.
    og.activate_from_sums(
        &input_sums[3 * hidden..],
        &recurrent_sums[3 * hidden..],
        Some(buffers.state),
        &mut gate_out[3 * hidden..],
    )?;
    buffers.activated.copy_from_slice(buffers.state);
    apply_activation_row(
        &mut ArrayViewMut1::from(&mut buffers.activated[..]),
        activation,
    )?;
    for j in 0..hidden {
        let v = buffers.activated[j] * gate_out[3 * hidden + j];
        out_row[j] = v;
        buffers.prev[j] = v;
    }
    Ok(())
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

    #[cfg(test)]
    fn activate_into(
        &self,
        input: ArrayView1<'_, f32>,
        prev: &Array1<f32>,
        state: Option<&Array1<f32>>,
        out: &mut Array1<f32>,
    ) -> Result<(), RhythmError> {
        if out.len() != self.bias.len() {
            return Err(RhythmError::Model(
                "gate output buffer length mismatch".to_string(),
            ));
        }
        vec_matmul(&input.to_owned(), &self.weights.view(), out);
        *out += &self.bias;
        if let (Some(ph), Some(st)) = (&self.peephole, state) {
            *out += &(st * ph);
        }
        let mut recurrent = Array1::zeros(out.len());
        vec_matmul(prev, &self.recurrent.view(), &mut recurrent);
        *out += &recurrent;
        apply_activation_row(&mut out.view_mut(), self.activation)?;
        Ok(())
    }

    /// Finishes a gate from its input and recurrent dot products, already
    /// summed, adding the terms in the order of madmom's `Gate.activate`:
    /// inputs plus bias, plus peephole, plus recurrent, then the activation.
    fn activate_from_sums(
        &self,
        input_sums: &[f32],
        recurrent_sums: &[f32],
        state: Option<&[f32]>,
        out: &mut [f32],
    ) -> Result<(), RhythmError> {
        for j in 0..out.len() {
            out[j] = input_sums[j] + self.bias[j];
        }
        if let (Some(ph), Some(st)) = (&self.peephole, state) {
            for j in 0..out.len() {
                out[j] += st[j] * ph[j];
            }
        }
        for j in 0..out.len() {
            out[j] += recurrent_sums[j];
        }
        apply_activation_row(&mut ArrayViewMut1::from(out), self.activation)
    }
}

/// `data · weights + bias`, each product a `numeric::dot`, as madmom's
/// `np.dot(data, weights) + bias`.
fn matmul_plus_bias(
    data: &Array2<f32>,
    weights: &ArrayView2<'_, f32>,
    bias: ArrayView1<'_, f32>,
) -> Result<Array2<f32>, RhythmError> {
    if weights.nrows() != data.ncols() || bias.len() != weights.ncols() {
        return Err(RhythmError::Model("layer shape mismatch".to_string()));
    }
    let columns: Vec<Vec<f32>> = weights.columns().into_iter().map(|c| c.to_vec()).collect();
    let mut out = Array2::<f32>::zeros((data.nrows(), columns.len()));
    for (row, mut out_row) in data.rows().into_iter().zip(out.rows_mut()) {
        let row = row.to_vec();
        for ((value, column), &b) in out_row.iter_mut().zip(&columns).zip(bias.iter()) {
            *value = numeric::dot(&row, column) + b;
        }
    }
    Ok(out)
}

/// `vector · matrix` into `out`, each element a `numeric::dot`.
fn vec_matmul(vector: &Array1<f32>, matrix: &ArrayView2<'_, f32>, out: &mut Array1<f32>) {
    let vector = vector.to_vec();
    for (value, column) in out.iter_mut().zip(matrix.columns()) {
        *value = numeric::dot(&vector, &column.to_vec());
    }
}

fn apply_activation(out: &mut Array2<f32>, act: Activation) -> Result<(), RhythmError> {
    for mut row in out.rows_mut() {
        apply_activation_row(&mut row, act)?;
    }
    Ok(())
}

fn apply_activation_row(
    row: &mut ArrayViewMut1<'_, f32>,
    act: Activation,
) -> Result<(), RhythmError> {
    match act {
        Activation::Linear => {}
        Activation::Tanh => row.mapv_inplace(numeric::tanh),
        Activation::Sigmoid => row.mapv_inplace(numeric::sigmoid),
        Activation::Relu => row.mapv_inplace(|v| if v > 0.0 { v } else { 0.0 }),
        Activation::Elu => row.mapv_inplace(|v| if v > 0.0 { v } else { numeric::exp(v) - 1.0 }),
        Activation::Softmax => {
            let max = row.iter().cloned().fold(f32::MIN, f32::max);
            let mut sum = 0.0;
            for v in row.iter_mut() {
                *v = numeric::exp(*v - max);
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

#[cfg(test)]
mod tests {
    use super::{average_network_outputs, lstm_forward, Activation, Gate};
    use ndarray::{Array1, Array2, Axis};

    fn random(len: usize, seed: &mut u64) -> Vec<f32> {
        (0..len)
            .map(|_| {
                *seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((*seed >> 33) as f32) / ((1u64 << 31) as f32) - 0.5
            })
            .collect()
    }

    fn gate(
        inputs: usize,
        hidden: usize,
        peephole: bool,
        activation: Activation,
        seed: &mut u64,
    ) -> Gate {
        Gate {
            weights: Array2::from_shape_vec((inputs, hidden), random(inputs * hidden, seed))
                .unwrap(),
            bias: Array1::from(random(hidden, seed)),
            recurrent: Array2::from_shape_vec((hidden, hidden), random(hidden * hidden, seed))
                .unwrap(),
            peephole: peephole.then(|| Array1::from(random(hidden, seed))),
            activation,
        }
    }

    /// A per-gate, per-frame LSTM to check `lstm_forward` against.
    fn lstm_reference(
        data: &Array2<f32>,
        gates: &[Gate; 4],
        activation: Activation,
    ) -> Array2<f32> {
        let [ig, fg, cg, og] = gates;
        let hidden = ig.bias.len();
        let mut out = Array2::<f32>::zeros((data.len_of(Axis(0)), hidden));
        let mut prev = Array1::<f32>::zeros(hidden);
        let mut state = Array1::<f32>::zeros(hidden);
        let mut ig_out = Array1::<f32>::zeros(hidden);
        let mut fg_out = Array1::<f32>::zeros(hidden);
        let mut cell_out = Array1::<f32>::zeros(hidden);
        let mut og_out = Array1::<f32>::zeros(hidden);
        for i in 0..data.len_of(Axis(0)) {
            let x = data.row(i);
            ig.activate_into(x, &prev, Some(&state), &mut ig_out)
                .unwrap();
            fg.activate_into(x, &prev, Some(&state), &mut fg_out)
                .unwrap();
            cg.activate_into(x, &prev, None, &mut cell_out).unwrap();
            for j in 0..hidden {
                state[j] = cell_out[j] * ig_out[j] + state[j] * fg_out[j];
            }
            og.activate_into(x, &prev, Some(&state), &mut og_out)
                .unwrap();
            let mut activated = state.clone();
            super::apply_activation_row(&mut activated.view_mut(), activation).unwrap();
            for j in 0..hidden {
                let v = activated[j] * og_out[j];
                out[(i, j)] = v;
                prev[j] = v;
            }
        }
        out
    }

    #[test]
    fn average_network_outputs_averages_and_drops_the_non_beat_column() {
        let a = Array2::from_shape_vec((1, 3), vec![0.1, 0.2, 0.4]).unwrap();
        let b = Array2::from_shape_vec((1, 3), vec![0.3, 0.4, 0.8]).unwrap();
        let activations = average_network_outputs(vec![a, b]).unwrap();
        assert_eq!(activations.beat, vec![(0.2f32 + 0.4) / 2.0]);
        assert_eq!(activations.downbeat, vec![(0.4f32 + 0.8) / 2.0]);
    }

    #[test]
    fn average_network_outputs_rejects_bad_inputs() {
        assert!(average_network_outputs(Vec::new()).is_err());
        let a = Array2::<f32>::zeros((2, 3));
        let b = Array2::<f32>::zeros((3, 3));
        assert!(average_network_outputs(vec![a, b]).is_err());
    }

    #[test]
    fn lstm_matches_reference_bit_for_bit() {
        let mut seed = 7u64;
        for &(inputs, hidden, peephole) in &[
            (314usize, 25usize, true),
            (50, 25, true),
            (3, 1, true),
            (50, 25, false),
        ] {
            let gates = [
                gate(inputs, hidden, peephole, Activation::Sigmoid, &mut seed),
                gate(inputs, hidden, peephole, Activation::Sigmoid, &mut seed),
                gate(inputs, hidden, peephole, Activation::Tanh, &mut seed),
                gate(inputs, hidden, peephole, Activation::Sigmoid, &mut seed),
            ];
            let frames = 300;
            let mut data =
                Array2::from_shape_vec((frames, inputs), random(frames * inputs, &mut seed))
                    .unwrap();
            let expected = lstm_reference(&data, &gates, Activation::Tanh);
            let actual = lstm_forward(&data, &gates, Activation::Tanh, false).unwrap();
            assert!(expected
                .iter()
                .zip(actual.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()));
            // Bidirectional layers feed time-reversed views.
            data.invert_axis(Axis(0));
            let expected = lstm_reference(&data, &gates, Activation::Tanh);
            let actual = lstm_forward(&data, &gates, Activation::Tanh, false).unwrap();
            assert!(expected
                .iter()
                .zip(actual.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }
}
