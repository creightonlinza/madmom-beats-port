// SPDX-License-Identifier: BSD-2-Clause
// Ported from madmom (BSD 2-Clause, Copyright (c) 2012-2014 JKU / OFAI). See NOTICE.

use crate::fft::RealFft;
use crate::numeric;
use crate::{CoreConfig, ProgressEvent, ProgressSink, ProgressStage, RhythmError};
use ndarray::{Array2, Axis};

#[derive(Debug, Clone)]
pub struct Features {
    /// Shape: frames x feature_dim
    pub data: Array2<f32>,
}

struct ResolutionPlan {
    frame_size: usize,
    window: Vec<f64>,
    /// Per band: the first bin and the weights up to its last non-zero bin.
    /// The triangular filters cover a few bins each, so summing only these
    /// skips almost all of the dense filterbank; the zeros it skips add
    /// nothing, so the sums come out bit-identical.
    band_weights: Vec<(usize, Vec<f32>)>,
    num_filters: usize,
    diff_frames: usize,
    positive_diffs: bool,
    column_offset: usize,
}

fn band_weights(filterbank: &Array2<f32>) -> Vec<(usize, Vec<f32>)> {
    filterbank
        .columns()
        .into_iter()
        .map(|column| {
            let first = column.iter().position(|&w| w != 0.0);
            let last = column.iter().rposition(|&w| w != 0.0);
            match (first, last) {
                (Some(first), Some(last)) => {
                    (first, column.slice(ndarray::s![first..=last]).to_vec())
                }
                _ => (0, Vec::new()),
            }
        })
        .collect()
}

/// Number of feature columns `compute_features` produces for this config.
pub fn feature_dim(config: &CoreConfig) -> Result<usize, RhythmError> {
    let feature_cfg = &config.feature;
    let mut total_cols = 0usize;
    for (&frame_size, &num_bands) in feature_cfg
        .frame_sizes
        .iter()
        .zip(feature_cfg.num_bands.iter())
    {
        let bin_freqs = bin_frequencies(feature_cfg.sample_rate, frame_size, frame_size / 2);
        let filterbank = build_log_filterbank(
            &bin_freqs,
            num_bands,
            feature_cfg.fmin,
            feature_cfg.fmax,
            true,
            true,
        )?;
        total_cols += filterbank.len_of(Axis(1)) * 2;
    }
    Ok(total_cols)
}

pub fn compute_features(
    samples: &[f32],
    config: &CoreConfig,
    progress: &mut dyn ProgressSink,
) -> Result<Features, RhythmError> {
    let feature_cfg = &config.feature;
    if samples.is_empty() {
        return Err(RhythmError::InvalidInput("samples empty".to_string()));
    }
    let log_progress = std::env::var("MADMOM_BEATS_PORT_PROGRESS").ok().as_deref() == Some("1");

    let num_frames = frame_count(samples.len(), feature_cfg.fps, feature_cfg.sample_rate)?;
    let mut plans = Vec::with_capacity(feature_cfg.frame_sizes.len());
    let mut total_cols = 0usize;

    for (&frame_size, &num_bands) in feature_cfg
        .frame_sizes
        .iter()
        .zip(feature_cfg.num_bands.iter())
    {
        let num_bins = frame_size / 2;
        let bin_freqs = bin_frequencies(feature_cfg.sample_rate, frame_size, num_bins);
        let filterbank = build_log_filterbank(
            &bin_freqs,
            num_bands,
            feature_cfg.fmin,
            feature_cfg.fmax,
            true,
            true,
        )?;
        let num_filters = filterbank.len_of(Axis(1));
        let hann = hann_window(frame_size);
        let diff_frames = diff_frames_from_window(
            &hann,
            feature_cfg.diff_ratio,
            feature_cfg.fps,
            feature_cfg.sample_rate,
        );
        plans.push(ResolutionPlan {
            frame_size,
            window: fft_window(&hann, feature_cfg.int16_scaling),
            band_weights: band_weights(&filterbank),
            num_filters,
            diff_frames,
            positive_diffs: feature_cfg.positive_diffs,
            column_offset: total_cols,
        });
        total_cols += num_filters * 2;
    }

    let mut output = Array2::<f32>::zeros((num_frames, total_cols));
    let total = plans.len() as f32;
    for (idx, plan) in plans.iter().enumerate() {
        if log_progress {
            println!(
                "features: frame_size {} -> {} frames, {} filters",
                plan.frame_size, num_frames, plan.num_filters
            );
        }
        fill_resolution_features(
            samples,
            feature_cfg.fps,
            feature_cfg.sample_rate,
            num_frames,
            plan,
            &mut output,
        )?;
        progress.on_progress(ProgressEvent {
            stage: ProgressStage::Features,
            progress: (idx as f32 + 1.0) / total,
        });
    }

    Ok(Features { data: output })
}

fn fill_resolution_features(
    samples: &[f32],
    fps: f32,
    sample_rate: u32,
    num_frames: usize,
    plan: &ResolutionPlan,
    output: &mut Array2<f32>,
) -> Result<(), RhythmError> {
    let hop_size = sample_rate as f64 / fps as f64;
    let frame_size = plan.frame_size;
    let num_bins = frame_size / 2;
    let num_filters = plan.num_filters;

    let mut fft = RealFft::new(frame_size);
    let mut frame = vec![0.0f64; frame_size];
    let mut spectrum = vec![(0.0f32, 0.0f32); num_bins];
    let mut magnitude = vec![0.0f32; num_bins];
    let mut log_row = vec![0.0f32; num_filters];
    let mut diff_row = vec![0.0f32; num_filters];

    let history_rows = plan.diff_frames.max(1);
    let mut history = vec![0.0f32; history_rows * num_filters];

    for frame_idx in 0..num_frames {
        feature_frame(
            samples,
            frame_idx,
            hop_size,
            plan,
            &mut fft,
            &mut FrameBuffers {
                frame: &mut frame,
                spectrum: &mut spectrum,
                magnitude: &mut magnitude,
                log_row: &mut log_row,
                diff_row: &mut diff_row,
                history: &mut history,
            },
            output,
        );
    }

    Ok(())
}

struct FrameBuffers<'a> {
    frame: &'a mut [f64],
    spectrum: &'a mut [(f32, f32)],
    magnitude: &'a mut [f32],
    log_row: &'a mut [f32],
    diff_row: &'a mut [f32],
    history: &'a mut [f32],
}

/// One frame's features. Kept out of line so WebAssembly engines see a
/// function called once per frame and switch it to optimized code early.
#[inline(never)]
fn feature_frame(
    samples: &[f32],
    frame_idx: usize,
    hop_size: f64,
    plan: &ResolutionPlan,
    fft: &mut RealFft,
    buffers: &mut FrameBuffers<'_>,
    output: &mut Array2<f32>,
) {
    let frame_size = plan.frame_size;
    let num_filters = plan.num_filters;
    let history_rows = plan.diff_frames.max(1);
    write_windowed_frame(
        samples,
        frame_idx,
        hop_size,
        frame_size,
        &plan.window,
        buffers.frame,
    );

    fft.process(buffers.frame, buffers.spectrum);
    for (magnitude, &(re, im)) in buffers.magnitude.iter_mut().zip(buffers.spectrum.iter()) {
        *magnitude = numeric::complex_abs(re, im);
    }

    for (log_value, (first, weights)) in buffers.log_row.iter_mut().zip(plan.band_weights.iter()) {
        let sum = numeric::dot(&buffers.magnitude[*first..*first + weights.len()], weights);
        *log_value = numeric::log10(sum + 1.0);
    }

    buffers.diff_row.fill(0.0);
    if frame_idx >= plan.diff_frames {
        let prev_slot = (frame_idx - plan.diff_frames) % history_rows;
        let prev_offset = prev_slot * num_filters;
        for (band, diff_value) in buffers.diff_row.iter_mut().enumerate().take(num_filters) {
            let delta = buffers.log_row[band] - buffers.history[prev_offset + band];
            *diff_value = if delta > 0.0 || !plan.positive_diffs {
                delta
            } else {
                0.0
            };
        }
    }

    let slot = frame_idx % history_rows;
    let slot_offset = slot * num_filters;
    buffers.history[slot_offset..slot_offset + num_filters].copy_from_slice(buffers.log_row);

    let log_start = plan.column_offset;
    let diff_start = log_start + num_filters;
    for (band, (&log_value, &diff_value)) in buffers
        .log_row
        .iter()
        .zip(buffers.diff_row.iter())
        .enumerate()
        .take(num_filters)
    {
        output[(frame_idx, log_start + band)] = log_value;
        output[(frame_idx, diff_start + band)] = diff_value;
    }
}

fn frame_count(sample_count: usize, fps: f32, sample_rate: u32) -> Result<usize, RhythmError> {
    if fps <= 0.0 {
        return Err(RhythmError::InvalidInput("fps must be > 0".to_string()));
    }
    let hop_size = sample_rate as f64 / fps as f64;
    Ok(((sample_count as f64) / hop_size).ceil() as usize)
}

fn write_windowed_frame(
    samples: &[f32],
    frame_idx: usize,
    hop_size: f64,
    frame_size: usize,
    window: &[f64],
    out: &mut [f64],
) {
    let ref_sample = (frame_idx as f64 * hop_size).floor() as isize;
    let start = ref_sample - (frame_size / 2) as isize;
    for i in 0..frame_size {
        let sample_idx = start + i as isize;
        let sample = if sample_idx < 0 || sample_idx >= samples.len() as isize {
            0.0
        } else {
            samples[sample_idx as usize]
        };
        out[i] = sample as f64 * window[i];
    }
}

fn diff_frames_from_window(window: &[f64], diff_ratio: f32, fps: f32, sample_rate: u32) -> usize {
    let hop_size = (sample_rate as f64) / fps as f64;
    let max_val = window.iter().copied().fold(f64::MIN, f64::max);
    let threshold = diff_ratio as f64 * max_val;
    let mut sample = 0usize;
    for (i, v) in window.iter().enumerate() {
        if *v > threshold {
            sample = i;
            break;
        }
    }
    let diff_samples = (window.len() as f64) / 2.0 - sample as f64;
    (diff_samples / hop_size).round_ties_even().max(1.0) as usize
}

/// `np.hanning(frame_size)`: `0.5 + 0.5 * cos(pi * k / (n - 1))` for
/// `k = 1 - n, 3 - n, ..., n - 1`.
fn hann_window(frame_size: usize) -> Vec<f64> {
    let n = frame_size as f64;
    (0..frame_size)
        .map(|i| {
            let k = 2.0 * i as f64 - (n - 1.0);
            0.5 + 0.5 * libm::cos(std::f64::consts::PI * k / (n - 1.0))
        })
        .collect()
}

/// madmom scales an int16 signal's window by `1 / i16::MAX`; samples arrive as `i16 / 32768`, so undo that.
fn fft_window(hann: &[f64], int16_scaling: bool) -> Vec<f64> {
    if int16_scaling {
        hann.iter().map(|w| w / i16::MAX as f64 * 32768.0).collect()
    } else {
        hann.to_vec()
    }
}

fn bin_frequencies(sample_rate: u32, fft_size: usize, num_bins: usize) -> Vec<f32> {
    let sr = sample_rate as f32;
    let n = fft_size as f32;
    (0..num_bins).map(|k| (k as f32) * sr / n).collect()
}

fn log_frequencies(bands_per_octave: usize, fmin: f32, fmax: f32, fref: f32) -> Vec<f32> {
    let bpo = bands_per_octave as f64;
    let fmin64 = fmin as f64;
    let fmax64 = fmax as f64;
    let fref64 = fref as f64;
    let left = libm::log2(fmin64 / fref64) * bpo;
    let right = libm::log2(fmax64 / fref64) * bpo;
    let left_i = left.floor() as i64;
    let right_i = right.ceil() as i64;

    let mut freqs = Vec::new();
    for i in left_i..right_i {
        let freq = fref64 * libm::pow(2.0, i as f64 / bpo);
        if freq >= fmin64 && freq <= fmax64 {
            freqs.push(freq as f32);
        }
    }
    freqs
}

fn frequencies_to_bins(
    frequencies: &[f32],
    bin_frequencies: &[f32],
    unique_bins: bool,
) -> Vec<usize> {
    let mut indices = Vec::with_capacity(frequencies.len());
    for &freq in frequencies {
        let idx = match bin_frequencies.binary_search_by(|v| v.partial_cmp(&freq).unwrap()) {
            Ok(i) => i,
            Err(i) => i,
        };
        let mut idx = idx.clamp(1, bin_frequencies.len() - 1);
        let left = bin_frequencies[idx - 1];
        let right = bin_frequencies[idx];
        if freq - left < right - freq {
            idx -= 1;
        }
        if !unique_bins || indices.last().copied() != Some(idx) {
            indices.push(idx);
        }
    }
    if unique_bins {
        indices.dedup();
    }
    indices
}

fn triangular_filter(start: usize, center: usize, stop: usize, norm: bool) -> Vec<f32> {
    let length = stop.saturating_sub(start);
    if length == 0 {
        return vec![];
    }
    let center_rel = center.saturating_sub(start);
    let fall_len = length.saturating_sub(center_rel);
    let mut data = linspace_open(0.0, 1.0, center_rel);
    data.extend(linspace_open(1.0, 0.0, fall_len));
    if norm {
        let sum = numeric::pairwise_sum(&data);
        if sum > 0.0 {
            for v in &mut data {
                *v /= sum;
            }
        }
    }
    data
}

/// `np.linspace(start, stop, num, endpoint=False)` cast to `float32`.
fn linspace_open(start: f64, stop: f64, num: usize) -> Vec<f32> {
    let step = (stop - start) / num as f64;
    (0..num).map(|i| (i as f64 * step + start) as f32).collect()
}

fn build_log_filterbank(
    bin_freqs: &[f32],
    num_bands: usize,
    fmin: f32,
    fmax: f32,
    norm_filters: bool,
    unique_filters: bool,
) -> Result<Array2<f32>, RhythmError> {
    let freqs = log_frequencies(num_bands, fmin, fmax, 440.0);
    let bins = frequencies_to_bins(&freqs, bin_freqs, unique_filters);
    if bins.len() < 3 {
        return Err(RhythmError::InvalidInput(
            "not enough bins for logarithmic filterbank".to_string(),
        ));
    }

    let mut filters = Vec::new();
    let mut idx = 0;
    while idx + 2 < bins.len() {
        let start = bins[idx];
        let mut center = bins[idx + 1];
        let mut stop = bins[idx + 2];
        if stop <= start {
            stop = start + 1;
            center = start;
        }
        if stop - start < 2 {
            center = start;
            stop = start + 1;
        }
        let data = triangular_filter(start, center, stop, norm_filters);
        filters.push((start, data));
        idx += 1;
    }

    let num_bins = bin_freqs.len();
    let num_filters = filters.len();
    let mut fb = Array2::<f32>::zeros((num_bins, num_filters));
    for (band, (start, filt)) in filters.into_iter().enumerate() {
        for (i, val) in filt.into_iter().enumerate() {
            let bin = start + i;
            if bin >= num_bins {
                break;
            }
            let current = fb[(bin, band)];
            fb[(bin, band)] = current.max(val);
        }
    }
    Ok(fb)
}

#[cfg(test)]
mod tests {
    use super::{
        band_weights, bin_frequencies, build_log_filterbank, compute_features, feature_dim,
        fft_window,
    };
    use crate::{CoreConfig, ProgressEvent};
    use ndarray::{Array2, Axis};

    fn default_filterbanks() -> Vec<Array2<f32>> {
        let feature = CoreConfig::default().feature;
        feature
            .frame_sizes
            .iter()
            .zip(feature.num_bands.iter())
            .map(|(&frame_size, &num_bands)| {
                let bin_freqs = bin_frequencies(feature.sample_rate, frame_size, frame_size / 2);
                build_log_filterbank(
                    &bin_freqs,
                    num_bands,
                    feature.fmin,
                    feature.fmax,
                    true,
                    true,
                )
                .unwrap()
            })
            .collect()
    }

    #[test]
    fn band_weights_keep_every_non_zero_weight() {
        for filterbank in default_filterbanks() {
            let bands = band_weights(&filterbank);
            assert_eq!(bands.len(), filterbank.len_of(Axis(1)));
            for (band, (first, weights)) in bands.iter().enumerate() {
                for (bin, &dense) in filterbank.column(band).iter().enumerate() {
                    let sparse = bin
                        .checked_sub(*first)
                        .and_then(|offset| weights.get(offset))
                        .copied()
                        .unwrap_or(0.0);
                    assert_eq!(dense.to_bits(), sparse.to_bits(), "band {band} bin {bin}");
                }
            }
        }
    }

    #[test]
    fn band_weights_treat_an_all_zero_band_as_empty() {
        let bands = band_weights(&Array2::<f32>::zeros((4, 1)));
        assert_eq!(bands, vec![(0, Vec::new())]);
    }

    #[test]
    fn feature_dim_matches_computed_features() {
        let config = CoreConfig::default();
        let samples = vec![0.25f32; 44_100];
        let mut sink = |_event: ProgressEvent| {};
        let features = compute_features(&samples, &config, &mut sink).unwrap();
        assert_eq!(features.data.len_of(Axis(1)), feature_dim(&config).unwrap());
    }

    #[test]
    fn int16_scaling_scales_window_by_32768_over_32767() {
        assert_eq!(fft_window(&[1.0, 0.5], false), vec![1.0, 0.5]);
        let scale = 32768.0 / 32767.0;
        assert_eq!(fft_window(&[1.0, 0.5], true), vec![scale, 0.5 * scale]);
    }
}
