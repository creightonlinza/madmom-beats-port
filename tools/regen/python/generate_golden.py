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


def frame_index(time_sec, fps, num_frames):
    idx = int(round(time_sec * fps))
    if idx < 0:
        return 0
    if idx >= num_frames:
        return num_frames - 1
    return idx


def process_file(path: Path, out_dir: Path, fps: float = 100.0, num_threads: int = 1):
    proc = RNNDownBeatProcessor(num_threads=num_threads)
    t0 = time.perf_counter()
    activations = proc(str(path)).astype(np.float32)
    t1 = time.perf_counter()
    print(f"  RNN activations: {t1 - t0:.2f}s", flush=True)

    dbn = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=fps, num_threads=num_threads)
    beats = dbn(activations)
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

    beat_events = []
    downbeat_events = []
    num_frames = activations.shape[0]

    for time_sec, beat_in_bar in beats:
        idx = frame_index(time_sec, fps, num_frames)
        beat_events.append(
            {
                "time_sec": float(time_sec),
                "confidence": float(activations[idx, 0]),
            }
        )
        if int(round(beat_in_bar)) == 1:
            downbeat_events.append(
                {
                    "time_sec": float(time_sec),
                    "beat_in_bar": int(round(beat_in_bar)),
                    "confidence": float(activations[idx, 1]),
                }
            )

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
        "beats": beat_events,
        "downbeats": downbeat_events,
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
