use crate::types::{
    ActivationOutput, BeatEvent, DecodedEvents, DownbeatEvent, ProgressEvent, ProgressSink,
    ProgressStage,
};
use crate::{CoreConfig, RhythmError};

pub fn decode(
    activations: &ActivationOutput,
    config: &CoreConfig,
    progress: &mut dyn ProgressSink,
) -> Result<DecodedEvents, RhythmError> {
    progress.on_progress(ProgressEvent {
        stage: ProgressStage::Dbn,
        progress: 0.0,
    });
    let fps = config.feature.fps;

    let mut act: Vec<[f32; 2]> = activations
        .beat
        .iter()
        .zip(activations.downbeat.iter())
        .map(|(b, d)| [*b, *d])
        .collect();

    let mut first = 0usize;
    if config.dbn.threshold > 0.0 {
        let (trimmed, offset) = threshold_activations(&act, config.dbn.threshold);
        act = trimmed;
        first = offset;
    }

    if act.is_empty() || !act.iter().any(|v| v[0] > 0.0 || v[1] > 0.0) {
        progress.on_progress(ProgressEvent {
            stage: ProgressStage::Dbn,
            progress: 1.0,
        });
        return Ok(DecodedEvents {
            beats: Vec::new(),
            downbeats: Vec::new(),
        });
    }

    let hmms = build_hmms(config)?;
    let mut best_logp = f64::NEG_INFINITY;
    let mut best_path: Vec<usize> = Vec::new();
    let mut best_hmm: Option<&HiddenMarkovModel> = None;

    for hmm in &hmms {
        let (path, logp) = hmm.viterbi(&act);
        if logp > best_logp {
            best_logp = logp;
            best_path = path;
            best_hmm = Some(hmm);
        }
    }

    let hmm = best_hmm.ok_or_else(|| RhythmError::Model("no HMM".to_string()))?;
    let st = &hmm.transition_model.state_space;
    let om = &hmm.observation_model;

    let mut beat_numbers = Vec::with_capacity(best_path.len());
    for &state in &best_path {
        let pos = st.state_positions[state];
        beat_numbers.push(pos.floor() as usize + 1);
    }

    let beat_indices = if config.dbn.correct {
        corrected_beats(&act, &best_path, om)?
    } else {
        transitions(&beat_numbers)
    };

    let mut beats = Vec::new();
    let mut downbeats = Vec::new();

    for &idx in &beat_indices {
        let frame = idx + first;
        let time_sec = (frame as f32) / fps;
        let beat_in_bar = beat_numbers[idx];
        let confidence = act[idx][0];
        beats.push(BeatEvent {
            time_sec,
            confidence,
        });
        if beat_in_bar == 1 {
            let db_conf = act[idx][1];
            downbeats.push(DownbeatEvent {
                time_sec,
                beat_in_bar,
                confidence: db_conf,
            });
        }
    }

    progress.on_progress(ProgressEvent {
        stage: ProgressStage::Dbn,
        progress: 1.0,
    });

    Ok(DecodedEvents { beats, downbeats })
}

fn threshold_activations(data: &[[f32; 2]], threshold: f32) -> (Vec<[f32; 2]>, usize) {
    let mut first = 0usize;
    let mut last = 0usize;
    let mut found = false;
    for (i, v) in data.iter().enumerate() {
        if v[0] >= threshold || v[1] >= threshold {
            if !found {
                first = i;
                found = true;
            }
            last = i + 1;
        }
    }
    if !found {
        return (Vec::new(), 0);
    }
    (data[first..last].to_vec(), first)
}

fn transitions(beat_numbers: &[usize]) -> Vec<usize> {
    let mut idx = Vec::new();
    for i in 1..beat_numbers.len() {
        if beat_numbers[i] != beat_numbers[i - 1] {
            idx.push(i);
        }
    }
    idx
}

fn corrected_beats(
    activations: &[[f32; 2]],
    path: &[usize],
    om: &ObservationModel,
) -> Result<Vec<usize>, RhythmError> {
    let mut beat_range = Vec::with_capacity(path.len());
    for &state in path {
        beat_range.push(om.pointers[state] >= 1);
    }
    if !beat_range.iter().any(|v| *v) {
        return Ok(Vec::new());
    }
    let mut idx = Vec::new();
    for i in 1..beat_range.len() {
        if beat_range[i] != beat_range[i - 1] {
            idx.push(i);
        }
    }
    if beat_range[0] {
        idx.insert(0, 0);
    }
    if *beat_range.last().unwrap() {
        idx.push(beat_range.len());
    }
    let mut beats = Vec::new();
    for pair in idx.chunks(2) {
        if pair.len() != 2 {
            continue;
        }
        let left = pair[0];
        let right = pair[1];
        let mut best_idx = left;
        let mut best_val = f32::MIN;
        for (i, act) in activations.iter().enumerate().take(right).skip(left) {
            let v = act[0].max(act[1]);
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        beats.push(best_idx);
    }
    Ok(beats)
}

fn build_hmms(config: &CoreConfig) -> Result<Vec<HiddenMarkovModel>, RhythmError> {
    let beats_per_bar = &config.dbn.beats_per_bar;
    let min_bpm = config.dbn.min_bpm;
    let max_bpm = config.dbn.max_bpm;
    let num_tempi = config.dbn.num_tempi;
    let transition_lambda = config.dbn.transition_lambda;

    let min_interval = 60.0 * config.feature.fps / max_bpm;
    let max_interval = 60.0 * config.feature.fps / min_bpm;

    let mut hmms = Vec::new();
    for &beats in beats_per_bar {
        let st = BarStateSpace::new(beats, min_interval, max_interval, Some(num_tempi))?;
        let tm = BarTransitionModel::new(&st, vec![Some(transition_lambda); st.num_beats])?;
        let om = ObservationModel::new(&st, config.dbn.observation_lambda)?;
        hmms.push(HiddenMarkovModel::new(tm, om));
    }
    Ok(hmms)
}

struct BeatStateSpace {
    num_states: usize,
    state_positions: Vec<f32>,
    state_intervals: Vec<usize>,
    first_states: Vec<usize>,
    last_states: Vec<usize>,
}

impl BeatStateSpace {
    fn new(
        min_interval: f32,
        max_interval: f32,
        num_intervals: Option<usize>,
    ) -> Result<Self, RhythmError> {
        let mut intervals: Vec<usize> =
            ((min_interval.round() as usize)..=(max_interval.round() as usize)).collect();
        if let Some(target) = num_intervals {
            if target < intervals.len() {
                let mut num_log = target;
                loop {
                    let mut candidate = logspace_intervals(min_interval, max_interval, num_log);
                    candidate.sort();
                    candidate.dedup();
                    if candidate.len() >= target {
                        intervals = candidate;
                        break;
                    }
                    num_log += 1;
                }
            }
        }
        let num_states: usize = intervals.iter().sum();
        let mut first_states = Vec::with_capacity(intervals.len());
        let mut last_states = Vec::with_capacity(intervals.len());
        let mut state_positions = vec![0.0; num_states];
        let mut state_intervals = vec![0usize; num_states];
        let mut idx = 0usize;
        for &interval in &intervals {
            first_states.push(idx);
            last_states.push(idx + interval - 1);
            for i in 0..interval {
                state_positions[idx + i] = (i as f32) / (interval as f32);
                state_intervals[idx + i] = interval;
            }
            idx += interval;
        }
        Ok(Self {
            num_states,
            state_positions,
            state_intervals,
            first_states,
            last_states,
        })
    }
}

#[derive(Clone)]
struct BarStateSpace {
    num_beats: usize,
    num_states: usize,
    state_positions: Vec<f32>,
    state_intervals: Vec<usize>,
    first_states: Vec<Vec<usize>>,
    last_states: Vec<Vec<usize>>,
}

impl BarStateSpace {
    fn new(
        num_beats: usize,
        min_interval: f32,
        max_interval: f32,
        num_intervals: Option<usize>,
    ) -> Result<Self, RhythmError> {
        let bss = BeatStateSpace::new(min_interval, max_interval, num_intervals)?;
        let mut state_positions = Vec::new();
        let mut state_intervals = Vec::new();
        let mut first_states = Vec::new();
        let mut last_states = Vec::new();
        let mut offset = 0usize;
        for beat in 0..num_beats {
            for &pos in &bss.state_positions {
                state_positions.push(pos + beat as f32);
            }
            state_intervals.extend_from_slice(&bss.state_intervals);
            first_states.push(bss.first_states.iter().map(|v| v + offset).collect());
            last_states.push(bss.last_states.iter().map(|v| v + offset).collect());
            offset += bss.num_states;
        }
        Ok(Self {
            num_beats,
            num_states: offset,
            state_positions,
            state_intervals,
            first_states,
            last_states,
        })
    }
}

#[derive(Clone)]
struct ObservationModel {
    pointers: Vec<usize>,
    observation_lambda: usize,
}

impl ObservationModel {
    fn new(state_space: &BarStateSpace, observation_lambda: usize) -> Result<Self, RhythmError> {
        let border = 1.0 / observation_lambda as f32;
        let mut pointers = vec![0usize; state_space.num_states];
        for (i, pos) in state_space.state_positions.iter().enumerate() {
            let beat_pos = pos % 1.0;
            if beat_pos < border {
                pointers[i] = 1;
            }
            if *pos < border {
                pointers[i] = 2;
            }
        }
        Ok(Self {
            pointers,
            observation_lambda,
        })
    }

    fn log_densities(&self, observations: &[[f32; 2]]) -> Vec<[f64; 3]> {
        let mut out = Vec::with_capacity(observations.len());
        for obs in observations {
            let beat = obs[0] as f64;
            let downbeat = obs[1] as f64;
            let sum = beat + downbeat;
            let none = (1.0 - sum) / (self.observation_lambda as f64 - 1.0);
            out.push([
                log_or_neg_inf(none),
                log_or_neg_inf(beat),
                log_or_neg_inf(downbeat),
            ]);
        }
        out
    }
}

#[derive(Clone)]
struct TransitionModel {
    states: Vec<usize>,
    pointers: Vec<usize>,
    log_probabilities: Vec<f64>,
    state_space: BarStateSpace,
}

struct BarTransitionModel {
    tm: TransitionModel,
}

impl BarTransitionModel {
    fn new(
        state_space: &BarStateSpace,
        transition_lambda: Vec<Option<f32>>,
    ) -> Result<Self, RhythmError> {
        if transition_lambda.len() != state_space.num_beats {
            return Err(RhythmError::InvalidInput(
                "transition_lambda length mismatch".to_string(),
            ));
        }
        let mut states = Vec::new();
        let mut prev_states = Vec::new();
        let mut probabilities = Vec::new();

        for state in 0..state_space.num_states {
            let is_first = state_space.first_states.iter().any(|v| v.contains(&state));
            if !is_first {
                states.push(state);
                prev_states.push(state - 1);
                probabilities.push(1.0);
            }
        }

        for (beat, lambda) in transition_lambda
            .iter()
            .enumerate()
            .take(state_space.num_beats)
        {
            let to_states = &state_space.first_states[beat];
            let from_states = &state_space.last_states
                [(beat + state_space.num_beats - 1) % state_space.num_beats];
            let from_int: Vec<usize> = from_states
                .iter()
                .map(|s| state_space.state_intervals[*s])
                .collect();
            let to_int: Vec<usize> = to_states
                .iter()
                .map(|s| state_space.state_intervals[*s])
                .collect();
            let prob = exponential_transition(&from_int, &to_int, *lambda);
            for (i_from, row) in prob.iter().enumerate() {
                for (i_to, p) in row.iter().enumerate() {
                    if *p > 0.0 {
                        states.push(to_states[i_to]);
                        prev_states.push(from_states[i_from]);
                        probabilities.push(*p);
                    }
                }
            }
        }

        let (states, pointers, log_probabilities) =
            make_sparse(&states, &prev_states, &probabilities)?;

        Ok(Self {
            tm: TransitionModel {
                states,
                pointers,
                log_probabilities,
                state_space: state_space.clone(),
            },
        })
    }
}

#[derive(Clone)]
struct HiddenMarkovModel {
    transition_model: TransitionModel,
    observation_model: ObservationModel,
    initial_distribution: Vec<f64>,
}

impl HiddenMarkovModel {
    fn new(transition_model: BarTransitionModel, observation_model: ObservationModel) -> Self {
        let num_states = transition_model.tm.pointers.len() - 1;
        let init = vec![1.0 / num_states as f64; num_states];
        Self {
            transition_model: transition_model.tm,
            observation_model,
            initial_distribution: init,
        }
    }

    fn viterbi(&self, observations: &[[f32; 2]]) -> (Vec<usize>, f64) {
        let num_states = self.transition_model.pointers.len() - 1;
        let num_frames = observations.len();
        let log_densities = self.observation_model.log_densities(observations);

        let mut current = vec![f64::NEG_INFINITY; num_states];
        let mut previous: Vec<f64> = self.initial_distribution.iter().map(|v| v.ln()).collect();
        let mut bt = vec![0usize; num_states * num_frames];

        for (frame, dens) in log_densities.iter().enumerate() {
            for state in 0..num_states {
                let density = dens[self.observation_model.pointers[state]];
                let mut best = f64::NEG_INFINITY;
                let mut best_prev = 0usize;
                let start = self.transition_model.pointers[state];
                let end = self.transition_model.pointers[state + 1];
                for idx in start..end {
                    let prev_state = self.transition_model.states[idx];
                    let trans = self.transition_model.log_probabilities[idx];
                    let score = previous[prev_state] + trans + density;
                    if score > best {
                        best = score;
                        best_prev = prev_state;
                    }
                }
                current[state] = best;
                bt[frame * num_states + state] = best_prev;
            }
            previous.clone_from_slice(&current);
        }

        let mut best_state = 0usize;
        let mut best_log = f64::NEG_INFINITY;
        for (i, v) in current.iter().enumerate() {
            if *v > best_log {
                best_log = *v;
                best_state = i;
            }
        }
        if best_log.is_infinite() && best_log.is_sign_negative() {
            return (Vec::new(), best_log);
        }

        let mut path = vec![0usize; num_frames];
        let mut state = best_state;
        for frame in (0..num_frames).rev() {
            path[frame] = state;
            state = bt[frame * num_states + state];
        }
        (path, best_log)
    }
}

fn exponential_transition(
    from_intervals: &[usize],
    to_intervals: &[usize],
    transition_lambda: Option<f32>,
) -> Vec<Vec<f32>> {
    if transition_lambda.is_none() {
        let mut prob = vec![vec![0.0; to_intervals.len()]; from_intervals.len()];
        for (i, from) in from_intervals.iter().enumerate() {
            for (j, to) in to_intervals.iter().enumerate() {
                if from == to {
                    prob[i][j] = 1.0;
                }
            }
        }
        return prob;
    }
    let lambda = transition_lambda.unwrap();
    let mut prob = vec![vec![0.0; to_intervals.len()]; from_intervals.len()];
    for (i, from) in from_intervals.iter().enumerate() {
        let mut sum = 0.0f32;
        for (j, to) in to_intervals.iter().enumerate() {
            let ratio = (*to as f32) / (*from as f32);
            let v = (-lambda * (ratio - 1.0).abs()).exp();
            prob[i][j] = v;
            sum += v;
        }
        if sum > 0.0 {
            for p in prob[i].iter_mut() {
                *p /= sum;
            }
        }
    }
    prob
}

type SparseTransition = (Vec<usize>, Vec<usize>, Vec<f64>);

fn make_sparse(
    states: &[usize],
    prev_states: &[usize],
    probabilities: &[f32],
) -> Result<SparseTransition, RhythmError> {
    let num_states = prev_states.iter().copied().max().unwrap_or(0) + 1;
    let mut sums = vec![0.0f32; num_states];
    for (&p, &prob) in prev_states.iter().zip(probabilities.iter()) {
        sums[p] += prob;
    }
    for s in sums {
        if (s - 1.0).abs() > 1e-3 {
            return Err(RhythmError::Model(
                "transition probabilities do not sum to 1".to_string(),
            ));
        }
    }

    let mut per_state: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_states];
    for ((&state, &prev), &prob) in states
        .iter()
        .zip(prev_states.iter())
        .zip(probabilities.iter())
    {
        per_state[state].push((prev, prob as f64));
    }
    let mut out_states = Vec::new();
    let mut pointers = Vec::with_capacity(num_states + 1);
    let mut log_probs = Vec::new();
    let mut offset = 0usize;
    pointers.push(0);
    for list in per_state {
        for (prev, prob) in list {
            out_states.push(prev);
            log_probs.push(log_or_neg_inf(prob));
            offset += 1;
        }
        pointers.push(offset);
    }
    Ok((out_states, pointers, log_probs))
}

fn log_or_neg_inf(v: f64) -> f64 {
    if v <= 0.0 {
        f64::NEG_INFINITY
    } else {
        v.ln()
    }
}

fn logspace_intervals(min_interval: f32, max_interval: f32, count: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(count);
    let min = min_interval as f64;
    let max = max_interval as f64;
    let log_min = min.log2();
    let log_max = max.log2();
    if count == 1 {
        out.push(min_interval.round() as usize);
        return out;
    }
    for i in 0..count {
        let t = i as f64 / (count as f64 - 1.0);
        let v = 2f64.powf(log_min + t * (log_max - log_min));
        out.push(v.round() as usize);
    }
    out
}
