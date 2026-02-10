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

## Parameters pinned for parity

- sample rate: `44100`
- fps: `100`
- frame sizes: `[1024, 2048, 4096]`
- log filterbank bands per resolution: `[3, 6, 12]`
- diff ratio: `0.5` (positive diffs)
- beats per bar: `[3, 4]`
- DBN defaults: madmom `DBNDownBeatTrackingProcessor` defaults (see `tools/regen/reference/madmom`)

## Artifact locations

- Models (exported from madmom):
  - `models/downbeats_blstm.json`
  - `models/downbeats_blstm_weights.npz`
- Golden outputs (generated locally, not committed):
  - `fixtures/golden/<fixture_name>/activations.npz`
  - `fixtures/golden/<fixture_name>/events.json`
