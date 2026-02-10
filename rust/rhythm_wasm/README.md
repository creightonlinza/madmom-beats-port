# rhythm_wasm

WASM wrapper around `rhythm_core` (built with `wasm-pack --target web`).

## Exported functions

- `analyze_json(samples, sampleRate, configJson?)`
- `analyze_json_with_model(samples, sampleRate, configJson?, modelJson, weightsNpz)`
- `analyze_json_with_model_progress(samples, sampleRate, configJson?, modelJson, weightsNpz, progressCb)`

All functions return a JS object matching the Rust `AnalysisOutput` shape
(activations + decoded events).

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
const weightsNpz = new Uint8Array(await (await fetch("./models/downbeats_blstm_weights.npz")).arrayBuffer());

const result = analyze_json_with_model_progress(
  samples,
  44100,
  null,
  modelJson,
  weightsNpz,
  (stage, progress) => postMessage({ stage, progress })
);
postMessage({ result });
```
