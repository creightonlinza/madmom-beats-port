use crate::{CoreConfig, ProgressEvent, ProgressSink, ProgressStage, RhythmError};
use ndarray::{s, Array2, Axis};
use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;

#[derive(Debug, Clone)]
pub struct Features {
    /// Shape: frames x feature_dim
    pub data: Array2<f32>,
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
    let log_progress = std::env::var("RHYTHM_PROGRESS").ok().as_deref() == Some("1");

    let mut stacked: Option<Array2<f32>> = None;
    let total = feature_cfg.frame_sizes.len() as f32;
    for (idx, (frame_size, num_bands)) in feature_cfg
        .frame_sizes
        .iter()
        .zip(feature_cfg.num_bands.iter())
        .enumerate()
    {
        let frames = frame_signal(
            samples,
            *frame_size,
            feature_cfg.fps,
            feature_cfg.sample_rate,
        )?;
        if log_progress {
            println!(
                "features: frame_size {} -> {} frames",
                frame_size,
                frames.len_of(Axis(0))
            );
        }
        let stft = stft(&frames)?;
        let spec = magnitude_spectrogram(&stft);
        let filtered = filtered_spectrogram(
            &spec,
            feature_cfg.sample_rate,
            *frame_size,
            *num_bands,
            feature_cfg.fmin,
            feature_cfg.fmax,
        )?;
        let log_spec = log_spectrogram(&filtered, 1.0, 1.0);
        let diff = diff_spectrogram(
            &log_spec,
            feature_cfg.diff_ratio,
            *frame_size,
            feature_cfg.fps,
            feature_cfg.sample_rate,
        );
        let stacked_local = hstack(&log_spec, &diff)?;

        stacked = Some(match stacked {
            None => stacked_local,
            Some(prev) => hstack(&prev, &stacked_local)?,
        });

        let pct = (idx as f32 + 1.0) / total;
        progress.on_progress(ProgressEvent {
            stage: ProgressStage::Features,
            progress: pct,
        });
    }

    let data = stacked.ok_or_else(|| RhythmError::InvalidInput("no features".to_string()))?;
    Ok(Features { data })
}

fn frame_signal(
    samples: &[f32],
    frame_size: usize,
    fps: f32,
    sample_rate: u32,
) -> Result<Array2<f32>, RhythmError> {
    if frame_size == 0 {
        return Err(RhythmError::InvalidInput("frame_size = 0".to_string()));
    }
    if fps <= 0.0 {
        return Err(RhythmError::InvalidInput("fps must be > 0".to_string()));
    }

    let hop_size = sample_rate as f32 / fps;
    let num_frames = ((samples.len() as f32) / hop_size).ceil() as usize;
    let mut frames = Array2::<f32>::zeros((num_frames, frame_size));
    let half = frame_size / 2;

    for i in 0..num_frames {
        let ref_sample = (i as f32 * hop_size).floor() as isize;
        let start = ref_sample - half as isize;
        for j in 0..frame_size {
            let idx = start + j as isize;
            let val = if idx < 0 || idx >= samples.len() as isize {
                0.0
            } else {
                samples[idx as usize]
            };
            frames[(i, j)] = val;
        }
    }
    Ok(frames)
}

fn hann_window(frame_size: usize) -> Vec<f32> {
    if frame_size == 0 {
        return vec![];
    }
    let n = frame_size as f32;
    (0..frame_size)
        .map(|i| {
            let x = i as f32;
            0.5 - 0.5 * (2.0 * std::f32::consts::PI * x / (n - 1.0)).cos()
        })
        .collect()
}

fn stft(frames: &Array2<f32>) -> Result<Array2<Complex32>, RhythmError> {
    let num_frames = frames.len_of(Axis(0));
    let frame_size = frames.len_of(Axis(1));
    let num_bins = frame_size / 2;

    let window = hann_window(frame_size);

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(frame_size);

    let mut output = Array2::<Complex32>::zeros((num_frames, num_bins));
    let mut buffer = vec![Complex32::new(0.0, 0.0); frame_size];

    for i in 0..num_frames {
        for j in 0..frame_size {
            let val = frames[(i, j)] * window[j];
            buffer[j] = Complex32::new(val, 0.0);
        }
        fft.process(&mut buffer);
        for k in 0..num_bins {
            output[(i, k)] = buffer[k];
        }
    }
    Ok(output)
}

fn magnitude_spectrogram(stft: &Array2<Complex32>) -> Array2<f32> {
    let mut out = Array2::<f32>::zeros(stft.raw_dim());
    for ((i, j), val) in stft.indexed_iter() {
        out[(i, j)] = val.norm();
    }
    out
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
    let left = (fmin64 / fref64).log2() * bpo;
    let right = (fmax64 / fref64).log2() * bpo;
    let left_i = left.floor() as i64;
    let right_i = right.ceil() as i64;

    let mut freqs = Vec::new();
    for i in left_i..right_i {
        let freq = fref64 * 2f64.powf(i as f64 / bpo);
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
    let mut data = vec![0.0f32; length];
    if center_rel > 0 {
        for (i, v) in data.iter_mut().enumerate().take(center_rel) {
            *v = (i as f32) / (center_rel as f32);
        }
    }
    let fall_len = length.saturating_sub(center_rel);
    if fall_len > 0 {
        for i in 0..fall_len {
            data[center_rel + i] = 1.0 - (i as f32) / (fall_len as f32);
        }
    }
    if norm {
        let sum: f32 = data.iter().sum();
        if sum > 0.0 {
            for v in &mut data {
                *v /= sum;
            }
        }
    }
    data
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

fn filtered_spectrogram(
    spec: &Array2<f32>,
    sample_rate: u32,
    fft_size: usize,
    num_bands: usize,
    fmin: f32,
    fmax: f32,
) -> Result<Array2<f32>, RhythmError> {
    let num_bins = spec.len_of(Axis(1));
    let bin_freqs = bin_frequencies(sample_rate, fft_size, num_bins);
    let fb = build_log_filterbank(&bin_freqs, num_bands, fmin, fmax, true, true)?;
    Ok(spec.dot(&fb))
}

fn log_spectrogram(spec: &Array2<f32>, mul: f32, add: f32) -> Array2<f32> {
    let mut out = spec.clone();
    if mul != 1.0 {
        out.mapv_inplace(|v| v * mul);
    }
    if add != 0.0 {
        out.mapv_inplace(|v| v + add);
    }
    out.mapv_inplace(|v| v.log10());
    out
}

fn diff_spectrogram(
    spec: &Array2<f32>,
    diff_ratio: f32,
    frame_size: usize,
    fps: f32,
    sample_rate: u32,
) -> Array2<f32> {
    let hop_size = (sample_rate as f32) / fps;
    let window = hann_window(frame_size);
    let max_val = window.iter().cloned().fold(f32::MIN, |a, b| a.max(b));
    let threshold = diff_ratio * max_val;
    let mut sample = 0usize;
    for (i, v) in window.iter().enumerate() {
        if *v > threshold {
            sample = i;
            break;
        }
    }
    let diff_samples = (window.len() as f32) / 2.0 - sample as f32;
    let diff_frames = (diff_samples / hop_size).round().max(1.0) as usize;

    let num_frames = spec.len_of(Axis(0));
    let num_bins = spec.len_of(Axis(1));
    let mut diff = Array2::<f32>::zeros((num_frames, num_bins));
    for i in diff_frames..num_frames {
        for j in 0..num_bins {
            let val = spec[(i, j)] - spec[(i - diff_frames, j)];
            diff[(i, j)] = if val > 0.0 { val } else { 0.0 };
        }
    }
    diff
}

fn hstack(a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>, RhythmError> {
    if a.len_of(Axis(0)) != b.len_of(Axis(0)) {
        return Err(RhythmError::InvalidInput(
            "frame count mismatch in hstack".to_string(),
        ));
    }
    let rows = a.len_of(Axis(0));
    let cols = a.len_of(Axis(1)) + b.len_of(Axis(1));
    let mut out = Array2::<f32>::zeros((rows, cols));
    out.slice_mut(s![.., 0..a.len_of(Axis(1))]).assign(a);
    out.slice_mut(s![.., a.len_of(Axis(1))..]).assign(b);
    Ok(out)
}
