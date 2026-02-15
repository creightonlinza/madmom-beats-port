# madmom-beats-port

Offline beat + downbeat detection ported from madmom into Rust for iOS/Android/WASM.

## Status

The Rust core (DSP, model inference, DBN decoding) matches the madmom reference
implementation under `tools/regen/reference/madmom` on golden fixtures.

## Build

### Rust (core + FFI)

```bash
cd rust
cargo build -p rhythm_core
cargo build -p rhythm_ffi
```

### WASM

```bash
cd rust/rhythm_wasm
wasm-pack build --target web
```

The WASM build outputs a distributable package under `rust/rhythm_wasm/pkg/`.

### iOS / Android

The C ABI in `rust/rhythm_ffi` builds as a `staticlib` and `cdylib`. Integrate via your platform build system (Xcode/Gradle). The exported API is declared in `rust/rhythm_ffi/include/rhythm.h` and documented in `rust/rhythm_ffi/README.md`.

## Usage

- FFI (C/Swift/JNI): `rust/rhythm_ffi/README.md`
- WASM (Web Worker/JS): `rust/rhythm_wasm/README.md`
- Public C header: `rust/rhythm_ffi/include/rhythm.h`

## Output contract

All platform APIs (Rust core, FFI, WASM) return the same shape:

```json
{
  "fps": 100,
  "beat_times": [0.2059, 0.645, 1.0222],
  "beat_numbers": [1, 2, 3],
  "beat_confidences": [0.83, 0.79, 0.81]
}
```

Invariants:

- `beat_times`, `beat_numbers`, `beat_confidences` have the same length.
- `beat_times` are strictly increasing.
- `beat_numbers` are 1-based in-bar beat positions.
- `beat_confidences` are in `[0, 1]`.

## Golden fixtures

Golden fixtures are generated locally under `fixtures/golden/` (ignored from
git). The golden test runs against whatever fixture directories are present
under `fixtures/golden/` and expects matching audio in `fixtures/audio/`.

To run parity tests in release mode, set:

```bash
cargo test -p rhythm_core --test golden --release
```

To run a single fixture by name:

```bash
FIXTURE=clap_snare_loop_carrai_pass cargo test -p rhythm_core --test golden --release
```

## Model export

Export downbeat BLSTM models to a Rust-friendly format:

```bash
python3 tools/regen/python/export_downbeats_model.py --out models
```

This writes:

- `models/downbeats_blstm.json`
- `models/downbeats_blstm_weights.npz`

Goldens are written locally under `fixtures/golden/<fixture_name>/` as:

- `activations.npz` (beat + downbeat arrays)
- `events.json` (authoritative `beats` list with `time_sec`, `beat_in_bar`, `confidence`)

## Licensing

- Code in this repo is permissively licensed (MIT OR Apache-2.0).
- Model assets under `models/` are **CC BY-NC-SA 4.0** (see `LICENSES/MODELS_CC_BY_NC_SA.txt`).
- Fixture audio is not tracked in git; keep local provenance if needed.
- Third-party notices are in `LICENSES/THIRD_PARTY.md`.
- Provenance of committed artifacts is in `PROVENANCE.md`.

## Regeneration (archived)

Regen tooling is archived under `tools/regen/` and not used by CI. See
`tools/regen/README.md` for commands and prerequisites.

## Versioning

Releases use semantic versioning with tags like `v1.0.0`. All artifacts
(Rust core, FFI, WASM, models) are versioned together and should be treated as
a single unit.
