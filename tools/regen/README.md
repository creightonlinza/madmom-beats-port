# Archived Regeneration Tools

These tools are **archived** and **not used by CI**. They exist only to
regenerate `models/` and `fixtures/golden/` when necessary.

## Prerequisites

- Python 3.10+ (local venv recommended)
- `numpy`
- `madmom` runtime dependencies from `tools/regen/reference/madmom/requirements.txt`
- Apply local madmom patches:
  - `git -C tools/regen/reference/madmom apply ../../patches/madmom-submodule.patch`

Optional (faster DBN decode):
- Build madmom C extensions from `tools/regen/reference/madmom`

## Regenerate models

```bash
./.venv/bin/python tools/regen/python/export_downbeats_model.py --out models
```

## Regenerate goldens

```bash
./.venv/bin/python tools/regen/python/generate_golden.py
```

## Known quirks / patches

- Patch file: `tools/regen/patches/madmom-submodule.patch`
  - Adds `tools/regen/reference/madmom/madmom/ml/hmm.py` as a pure‑Python
    fallback to avoid Cython build requirements.
  - Adjusts `tools/regen/reference/madmom/madmom/__init__.py` to tolerate
    missing package metadata when running from a submodule checkout.

If you update or remove these patches, re‑validate parity and update
`PROVENANCE.md`.
