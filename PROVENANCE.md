# Provenance

This document records how the committed model artifacts and local golden
artifacts were produced.

## Source repositories

- `madmom` source:
  - Submodule: `tools/regen/reference/madmom`
  - Version: `0.17.dev0` (from `tools/regen/reference/madmom/setup.py`)
  - Commit: `27f032e8947204902c675e5e341a3faf5dc86dae`
- `madmom_models` source:
  - Submodule: `tools/regen/reference/madmom_models`
  - Commit: `7e3dc1b0cad499792767074d03c38b194b9b0a79`

## Local patches

- `tools/regen/patches/madmom-submodule.patch`

## Artifact generation commands

From the repo root:

```bash
./.venv/bin/python tools/regen/python/export_downbeats_model.py --out models
./.venv/bin/python tools/regen/python/generate_golden.py
```

## Golden environment

The goldens come from macOS 26.6 arm64 (Apple libm, Accelerate), Python
3.14.8, numpy 2.4.2, scipy 1.17.0, madmom with its compiled Cython extensions
and the local patch. An independent run of unmodified upstream madmom (Python
3.11, numpy 2.2.3, scipy 1.13.1) on the same machine gives identical
activations. madmom's own results depend on the platform's math library and
BLAS, so regenerate the goldens after a macOS update.

The port computes the same bits on each platform checked so far (macOS arm64,
x86-64, WASM; CI runs the bit-exact snapshot tests on Linux x86-64). It follows madmom's
operation order and precision, but rounds multiply-adds in two steps and
takes `exp`, `log` and the FFT from portable code, where madmom uses fused
multiply-adds, Apple libm and pocketfft. On 28 local tracks (13,387 beats; the
4 fixtures plus commercial music that is not in the repo), beat
counts and bar positions match madmom exactly, 64 beat times differ by at most
0.14 ms, and confidences by at most 0.0009.

## Parameters pinned for parity

- sample rate: `44100`
- fps: `100`
- frame sizes: `[1024, 2048, 4096]`
- log filterbank bands per resolution: `[3, 6, 12]`
- diff ratio: `0.5` (positive diffs)
- `int16_scaling`: `true` for 16-bit input passed as `i16 / 32768` (stereo
  downmixed to the integer channel mean, truncated, as madmom does), `false`
  for float input
- beats per bar: `[3, 4]`
- DBN defaults: madmom `DBNDownBeatTrackingProcessor` defaults (see `tools/regen/reference/madmom`)

## Artifact locations

- Models (exported from madmom):
  - `models/downbeats_blstm.json`
  - `models/downbeats_blstm_weights.npz`
- Golden outputs (generated locally, not committed):
  - `fixtures/golden/<fixture_name>/activations.npz`
  - `fixtures/golden/<fixture_name>/events.json`
