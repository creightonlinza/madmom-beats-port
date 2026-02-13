# rhythm_wasm

WASM wrapper around `rhythm_core` (built with `wasm-pack --target web`).

## Exported functions

- `analyze_json(samples, sampleRate, configJson?)`
- `analyze_json_with_model(samples, sampleRate, configJson?, modelJson, weightsNpz)`
- `analyze_json_with_model_progress(samples, sampleRate, configJson?, modelJson, weightsNpz, progressCb)`

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

## Model assets

The WASM build does not embed model files. Provide:

- `models/downbeats_blstm.json`
- `models/downbeats_blstm_weights.npz` (binary data)

Use `analyze_json_with_model*` and pass the model JSON string and NPZ bytes.

## Worker usage (minimal)

```js
import init, { analyze_json_with_model_progress } from "./rhythm_wasm.js";

await init();
const modelJson = await (await fetch("./models/downbeats_blstm.json")).text();
const weightsNpz = new Uint8Array(
  await (await fetch("./models/downbeats_blstm_weights.npz")).arrayBuffer(),
);

const result = analyze_json_with_model_progress(
  samples,
  44100,
  null,
  modelJson,
  weightsNpz,
  (stage, progress) => postMessage({ stage, progress }),
);
postMessage({ result });
```
