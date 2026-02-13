#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import time

ROOT = Path(__file__).resolve().parents[3]
REF_MADMOM = ROOT / "tools" / "regen" / "reference" / "madmom"
if str(REF_MADMOM) not in sys.path:
    sys.path.insert(0, str(REF_MADMOM))

import madmom  # noqa: E402
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor  # noqa: E402


def refine_beat_indices(indices: np.ndarray, activations: np.ndarray):
    energy = activations.sum(axis=1) if activations.ndim > 1 else activations
    count = energy.shape[0]
    refined = np.empty(len(indices), dtype=float)
    peaks = np.empty(len(indices), dtype=int)
    for i, idx in enumerate(indices):
        idx = int(idx)
        if idx <= 0 or idx >= count - 1:
            refined[i] = float(idx)
            peaks[i] = idx
            continue
        left = max(idx - 1, 0)
        right = min(idx + 1, count - 1)
        peak = left + int(np.argmax(energy[left:right + 1]))
        peaks[i] = peak
        if peak <= 0 or peak >= count - 1:
            refined[i] = float(peak)
            continue
        y1, y2, y3 = energy[peak - 1], energy[peak], energy[peak + 1]
        denom = y1 - 2 * y2 + y3
        if abs(float(denom)) < 1e-12:
            refined[i] = float(peak)
            continue
        delta = 0.5 * (y1 - y3) / denom
        refined[i] = float(peak) + float(np.clip(delta, -0.5, 0.5))
    return refined, peaks


def downbeats_with_refine_and_conf(proc, activations: np.ndarray) -> np.ndarray:
    first = 0
    if proc.threshold:
        idx = np.nonzero(activations >= proc.threshold)[0]
        if idx.any():
            first = max(first, int(np.min(idx)))
            last = min(len(activations), int(np.max(idx)) + 1)
        else:
            last = first
        activations = activations[first:last]
    if not activations.any():
        return np.empty((0, 3), dtype=np.float32)

    results = [hmm.viterbi(activations) for hmm in proc.hmms]
    best = int(np.argmax([float(r[1]) for r in results]))
    path, _ = results[best]
    st = proc.hmms[best].transition_model.state_space
    om = proc.hmms[best].observation_model
    positions = st.state_positions[path]
    beat_numbers = positions.astype(int) + 1

    if proc.correct:
        beats = np.empty(0, dtype=int)
        beat_range = om.pointers[path] >= 1
        idx = np.nonzero(np.diff(beat_range.astype(int)))[0] + 1
        if beat_range[0]:
            idx = np.r_[0, idx]
        if beat_range[-1]:
            idx = np.r_[idx, beat_range.size]
        if idx.any():
            for left, right in idx.reshape((-1, 2)):
                peak = np.argmax(activations[left:right]) // 2 + left
                beats = np.hstack((beats, peak))
    else:
        beats = np.nonzero(np.diff(beat_numbers))[0] + 1
    if beats.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    refined, peaks = refine_beat_indices(beats, activations)
    times = (refined + float(first)) / float(proc.fps)
    energy = activations.sum(axis=1) if activations.ndim > 1 else activations
    min_e = float(energy.min()) if energy.size else 0.0
    max_e = float(energy.max()) if energy.size else 0.0
    if max_e - min_e < 1e-6:
        conf = np.full_like(refined, 0.5, dtype=float)
    else:
        conf = (energy[peaks] - min_e) / (max_e - min_e)
        conf = np.clip(conf, 0.0, 1.0)

    beats_out = np.vstack((times, beat_numbers[beats], conf)).T.astype(np.float32)
    order = np.argsort(beats_out[:, 0], kind="stable")
    beats_out = beats_out[order]
    if beats_out.shape[0] > 1:
        keep = np.ones(beats_out.shape[0], dtype=bool)
        keep[1:] = np.diff(beats_out[:, 0]) > 0.0
        beats_out = beats_out[keep]
    return beats_out


def process_file(path: Path, out_dir: Path, fps: float = 100.0, num_threads: int = 1):
    proc = RNNDownBeatProcessor(fps=fps, num_threads=num_threads)
    t0 = time.perf_counter()
    activations = proc(str(path)).astype(np.float32)
    t1 = time.perf_counter()
    print(f"  RNN activations: {t1 - t0:.2f}s", flush=True)

    dbn = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=fps, num_threads=num_threads)
    beats_arr = downbeats_with_refine_and_conf(dbn, activations)
    t2 = time.perf_counter()
    print(f"  DBN decode: {t2 - t1:.2f}s", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_dir / "activations.npz",
        beat=activations[:, 0],
        downbeat=activations[:, 1],
        fps=np.array([fps], dtype=np.float32),
        sample_rate=np.array([44100], dtype=np.int32),
    )

    beats = []
    for time_sec, beat_in_bar, confidence in beats_arr:
        beat_in_bar = int(round(float(beat_in_bar)))
        confidence = float(np.clip(confidence, 0.0, 1.0))
        time_sec = float(time_sec)
        beat_event = {
            "time_sec": time_sec,
            "beat_in_bar": beat_in_bar,
            "confidence": confidence,
        }
        beats.append(beat_event)

    metadata = {
        "madmom_version": getattr(madmom, "__version__", "unknown"),
        "fps": fps,
        "sample_rate": 44100,
        "frame_sizes": [1024, 2048, 4096],
        "num_bands": [3, 6, 12],
        "diff_ratio": 0.5,
        "positive_diffs": True,
        "beats_per_bar": [3, 4],
    }

    payload = {
        "metadata": metadata,
        "beats": beats,
    }

    with open(out_dir / "events.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixtures",
        default=str(ROOT / "fixtures" / "audio"),
        help="Directory with input wav files",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "fixtures" / "golden"),
        help="Output directory",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=int(os.environ.get("MADMOM_NUM_THREADS", "4")),
        help="Number of threads for madmom processors",
    )
    args = parser.parse_args()

    fixtures = Path(args.fixtures)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    for wav in sorted(fixtures.glob("*.wav")):
        name = wav.stem
        out_dir = out_root / name
        print(f"Processing {wav} -> {out_dir}")
        process_file(wav, out_dir, num_threads=max(1, args.threads))


if __name__ == "__main__":
    main()
