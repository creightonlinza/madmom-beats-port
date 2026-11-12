# madmom_beats_port_wasm

WASM wrapper around `madmom_beats_port_core`.

## Build

```bash
sh build_pkg.sh
```

This runs `wasm-pack build --target web`, copies `LICENSE` and `NOTICE` into
the generated `pkg/` distribution, and includes them in the package file
allowlist.

## Exported functions

- `default_config_json()`
- `validate_config_json(configJson?)`
- `analyze_json(samples, sampleRate, configJson?)`
- `analyze_json_with_model(samples, sampleRate, configJson?, modelJson, weightsNpz)`
- `analyze_json_with_model_progress(samples, sampleRate, configJson?, modelJson, weightsNpz, progressCb)`
- Staged analysis for several workers: `feature_columns_json`,
  `extract_features_json`, `BeatNetworks`, `decode_beats` (see below)

Versioned JSON schemas:

- `docs/schemas/config.v1.schema.json`
- `docs/schemas/analysis-output.v1.schema.json`
- `docs/schemas/config.v1.example.json`

For 16-bit audio passed as `i16 / 32768`, set `feature.int16_scaling` to `true`
to match madmom on the same 16-bit file; downmix stereo to the integer channel
mean, truncated, as madmom does.

All functions return beat arrays:

```json
{
  "fps": 100,
  "beat_times": [0.2059, 0.645, 1.0222],
  "beat_numbers": [1, 2, 3],
  "beat_confidences": [0.83, 0.79, 0.81]
}
```

Output invariants:

- `beat_times`, `beat_numbers`, `beat_confidences` have the same length
- `beat_times` are strictly increasing
- `beat_numbers` are 1-based beat positions in bar
- `beat_confidences` are clamped to `[0, 1]`

`validate_config_json(configJson?)` returns:

- `null` on success
- error object on failure:
  - `code`: `CONFIG_PARSE_ERROR` or `CONFIG_VALIDATION_ERROR`
  - `message`
  - `path` (if known)
  - `context` (line/column for parse errors)

## Model assets

The WASM build does not embed model files. Provide:

- `models/downbeats_blstm.json`
- `models/downbeats_blstm_weights.npz` (binary data)

Use `analyze_json_with_model*` and pass the model JSON string and NPZ bytes.

## Worker usage (minimal)

```js
import init, {
  analyze_json_with_model_progress,
  default_config_json,
  validate_config_json,
} from "./madmom_beats_port_wasm.js";

await init();
const modelJson = await (await fetch("./models/downbeats_blstm.json")).text();
const weightsNpz = new Uint8Array(
  await (await fetch("./models/downbeats_blstm_weights.npz")).arrayBuffer(),
);
const defaultConfig = default_config_json();
const maybeValidationError = validate_config_json(defaultConfig);
if (maybeValidationError !== null) {
  throw new Error(JSON.stringify(maybeValidationError));
}

const result = analyze_json_with_model_progress(
  samples,
  44100,
  defaultConfig,
  modelJson,
  weightsNpz,
  (stage, progress) => postMessage({ stage, progress }),
);
postMessage({ result });
```

Progress stage IDs:

- `0`: features
- `1`: inference
- `2`: DBN decode

Cancellation:

- No explicit cancellation API yet.
- Run analysis in a Web Worker and terminate/replace the worker for cancellation.

## Staged analysis (several workers)

WASM runs single-threaded, but the analysis splits into independent steps
that can run on separate Web Workers, each with its own instance of this
module. The result is bit-identical to `analyze_json_with_model*` however the
steps are spread:

- `feature_columns_json(configJson?)`: feature columns per frame.
- `extract_features_json(samples, sampleRate, configJson?)`: the features as a
  `Float32Array`, frames x columns in row order.
- `new BeatNetworks(modelJson, weightsNpz)`: the model, loaded once per worker.
  - `count()`: number of networks in the ensemble (8 for the default model).
  - `set_features(features, columns)`: keeps the features for `run`.
  - `run(index)`: network `index`'s output as a `Float32Array`, frames x
    outputs in row order.
- `decode_beats(outputs, networks, columns, configJson?, progressCb?)`: decodes
  the beats from all networks' outputs concatenated in network order. Returns
  `{ log_probability, output }`; `output` has the usual beat arrays, or both
  are `undefined` where no bar length fits the track. A track with no activity
  above the threshold gives empty beat arrays and a `log_probability` of
  `-Infinity`.

Each bar length in `dbn.beats_per_bar` can also be decoded on its own worker:
run `decode_beats` once per length with `beats_per_bar: [n]`, then keep the
first result (in `beats_per_bar` order) with the highest `log_probability`.
That is exactly what the single-call decode picks.

```js
// One worker: the features.
const columns = feature_columns_json(config);
const features = extract_features_json(samples, 44100, config);
const frames = features.length / columns;

// Each network worker: a copy of the features and some network indices.
const nets = new BeatNetworks(modelJson, weightsNpz);
nets.set_features(features, columns);
const outputs = indices.map((index) => nets.run(index));

// Coordinator: every network's output, concatenated in network order.
const all = concatFloat32(outputsInNetworkOrder);
const outputColumns = outputsInNetworkOrder[0].length / frames;
const { output } = decode_beats(all, networkCount, outputColumns, config);
```

The networks dominate the run time, so spreading them over a few workers
gives most of the gain.

Workers trade memory for that speed. Each one holds its own copy of the module
and whatever it is given, and its WASM memory doesn't shrink until the worker
is terminated. For a 3:40 track, a single `analyze_json_with_model` call peaks
at about 115 MB of WASM memory, while 8 workers total about 560 MB, and 4 about
280 MB. Use fewer workers where memory matters, and terminate them when done.

## Licensing / redistribution

When you distribute the generated `pkg/` (the `.wasm` + JS glue), include
`LICENSE` and `NOTICE`. The helper script above copies both files into `pkg/`.
The source code is BSD 2-Clause (NumPy-derived parts BSD 3-Clause), and binary
redistributions must reproduce the
notices in `NOTICE` (madmom, NumPy and the `libm` crate).

If you distribute model files alongside `pkg/`, also include
`LICENSES/MODELS_CC_BY_NC_SA.txt` from the repo root and mark that
model-bearing distribution as CC BY-NC-SA 4.0 / non-commercial.
