#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import importlib.util
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REF_MADMOM = ROOT / "tools" / "regen" / "reference" / "madmom" / "madmom"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _bootstrap_madmom_minimal():
    # Create minimal package structure to satisfy pickle imports without
    # executing madmom/__init__.py (which imports compiled HMM).
    for name in ["madmom", "madmom.ml", "madmom.ml.nn"]:
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module

    _load_module("madmom.utils", REF_MADMOM / "utils" / "__init__.py")

    # Minimal io with open_file to support Processor.load.
    io_module = types.ModuleType("madmom.io")
    sys.modules["madmom.io"] = io_module
    utils = sys.modules["madmom.utils"]

    def open_file(filename, mode="r"):
        import contextlib
        import io as _io

        @contextlib.contextmanager
        def _ctx():
            if isinstance(filename, utils.string_types):
                f = fid = _io.open(filename, mode)
            else:
                f = filename
                fid = None
            try:
                yield f
            finally:
                if fid:
                    fid.close()

        return _ctx()

    io_module.open_file = open_file

    _load_module("madmom.processors", REF_MADMOM / "processors.py")
    _load_module("madmom.ml.nn", REF_MADMOM / "ml" / "nn" / "__init__.py")
    _load_module("madmom.ml.nn.activations", REF_MADMOM / "ml" / "nn" / "activations.py")
    _load_module("madmom.ml.nn.layers", REF_MADMOM / "ml" / "nn" / "layers.py")


_bootstrap_madmom_minimal()

from madmom.processors import Processor  # noqa: E402
from madmom.ml.nn import layers as nn_layers  # noqa: E402


def activation_name(fn):
    if fn is None:
        return "linear"
    name = getattr(fn, "__name__", None)
    if name in {"tanh", "sigmoid", "relu", "softmax", "linear", "elu"}:
        return name
    raise ValueError(f"Unsupported activation: {fn}")


def export_gate(prefix, gate, arrays, layer_meta):
    arrays[f"{prefix}_weights"] = gate.weights.astype(np.float32)
    arrays[f"{prefix}_bias"] = gate.bias.astype(np.float32)
    arrays[f"{prefix}_recurrent"] = gate.recurrent_weights.astype(np.float32)
    if getattr(gate, "peephole_weights", None) is not None:
        arrays[f"{prefix}_peephole"] = gate.peephole_weights.astype(np.float32)
        peephole = f"{prefix}_peephole"
    else:
        peephole = None
    layer_meta["weights"] = f"{prefix}_weights"
    layer_meta["bias"] = f"{prefix}_bias"
    layer_meta["recurrent_weights"] = f"{prefix}_recurrent"
    layer_meta["peephole_weights"] = peephole
    layer_meta["activation"] = activation_name(gate.activation_fn)


def export_layer(layer, arrays, prefix):
    if isinstance(layer, nn_layers.BidirectionalLayer):
        return {
            "type": "bidirectional",
            "fwd": export_layer(layer.fwd_layer, arrays, f"{prefix}_fwd"),
            "bwd": export_layer(layer.bwd_layer, arrays, f"{prefix}_bwd"),
        }
    if isinstance(layer, nn_layers.LSTMLayer):
        meta = {"type": "lstm", "activation": activation_name(layer.activation_fn)}
        meta["input_gate"] = {}
        export_gate(f"{prefix}_ig", layer.input_gate, arrays, meta["input_gate"])
        meta["forget_gate"] = {}
        export_gate(f"{prefix}_fg", layer.forget_gate, arrays, meta["forget_gate"])
        meta["cell"] = {}
        export_gate(f"{prefix}_cell", layer.cell, arrays, meta["cell"])
        meta["output_gate"] = {}
        export_gate(f"{prefix}_og", layer.output_gate, arrays, meta["output_gate"])
        return meta
    if isinstance(layer, nn_layers.FeedForwardLayer):
        arrays[f"{prefix}_weights"] = layer.weights.astype(np.float32)
        arrays[f"{prefix}_bias"] = layer.bias.astype(np.float32)
        return {
            "type": "feedforward",
            "weights": f"{prefix}_weights",
            "bias": f"{prefix}_bias",
            "activation": activation_name(layer.activation_fn),
        }
    if isinstance(layer, nn_layers.RecurrentLayer):
        arrays[f"{prefix}_weights"] = layer.weights.astype(np.float32)
        arrays[f"{prefix}_bias"] = layer.bias.astype(np.float32)
        arrays[f"{prefix}_recurrent"] = layer.recurrent_weights.astype(np.float32)
        return {
            "type": "recurrent",
            "weights": f"{prefix}_weights",
            "bias": f"{prefix}_bias",
            "recurrent_weights": f"{prefix}_recurrent",
            "activation": activation_name(layer.activation_fn),
        }

    raise ValueError(f"Unsupported layer type: {type(layer)}")


def export_model(model_paths, out_dir: Path):
    arrays = {}
    networks_meta = []

    for net_idx, model_path in enumerate(model_paths):
        nn = Processor.load(model_path)
        layers_meta = []
        for layer_idx, layer in enumerate(nn.layers):
            prefix = f"net{net_idx}_layer{layer_idx}"
            layers_meta.append(export_layer(layer, arrays, prefix))
        networks_meta.append({"layers": layers_meta})

    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "downbeats_blstm_weights.npz"
    json_path = out_dir / "downbeats_blstm.json"

    np.savez(weights_path, **arrays)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "networks": networks_meta}, f, indent=2)

    print(f"Wrote {json_path}")
    print(f"Wrote {weights_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        help="Model pickle files (downbeats_blstm_*.pkl)",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "models"),
        help="Output directory",
    )
    args = parser.parse_args()

    if not args.models:
        default_glob = (
            ROOT
            / "tools"
            / "regen"
            / "reference"
            / "madmom_models"
            / "downbeats"
            / "2016"
        )
        args.models = [str(p) for p in sorted(default_glob.glob("downbeats_blstm_*.pkl"))]

    export_model(args.models, Path(args.out))


if __name__ == "__main__":
    main()
