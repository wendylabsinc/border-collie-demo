"""Fold and export trained PyTorch weights for the native MAX graph."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import (
    ARCHITECTURE_SHA256,
    DEFAULT_SPEC,
    ConvNorm,
    TrainableFruitDetector,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hwio(weight: torch.Tensor) -> np.ndarray:
    """Convert PyTorch OIHW filters to MAX's grouped-convolution HWIO layout."""

    return np.ascontiguousarray(
        weight.detach().cpu().permute(2, 3, 1, 0).numpy()
    )


def fold_conv_norm(layer: ConvNorm) -> tuple[np.ndarray, np.ndarray]:
    """Fold an evaluation-mode BatchNorm into a convolution and emit HWIO."""

    norm = layer.norm
    scale = norm.weight.detach() / torch.sqrt(
        norm.running_var.detach() + norm.eps
    )
    weight = layer.conv.weight.detach() * scale.reshape(-1, 1, 1, 1)
    if layer.conv.bias is None:
        conv_bias = torch.zeros_like(norm.running_mean)
    else:
        conv_bias = layer.conv.bias.detach()
    bias = norm.bias.detach() + (conv_bias - norm.running_mean.detach()) * scale
    return _hwio(weight), np.ascontiguousarray(bias.cpu().numpy())


def export_checkpoint(
    checkpoint_path: Path,
    weights_path: Path,
    manifest_path: Path,
    *,
    dtype: np.dtype[Any] = np.dtype(np.float16),
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["architecture_sha256"] != ARCHITECTURE_SHA256:
        raise RuntimeError("checkpoint does not match the current MAX architecture")
    if tuple(checkpoint["classes"]) != DEFAULT_SPEC.classes:
        raise RuntimeError("checkpoint class map does not match the deployment contract")
    model = TrainableFruitDetector(DEFAULT_SPEC)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    foldable = dict(model.named_foldable_layers())
    registry: dict[str, np.ndarray] = {}
    for name, convolution in model.named_export_layers():
        if name in foldable:
            weight, bias = fold_conv_norm(foldable[name])
        else:
            weight = _hwio(convolution.weight)
            if convolution.bias is None:
                bias = np.zeros(convolution.out_channels, dtype=np.float32)
            else:
                bias = np.ascontiguousarray(convolution.bias.detach().cpu().numpy())
        registry[f"{name}.weight"] = weight.astype(dtype, copy=False)
        registry[f"{name}.bias"] = bias.astype(dtype, copy=False)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(weights_path, **registry)
    manifest = {
        "schema_version": 1,
        "architecture": DEFAULT_SPEC.manifest(),
        "architecture_sha256": ARCHITECTURE_SHA256,
        "classes": list(DEFAULT_SPEC.classes),
        "resolution": int(checkpoint["input_size"]),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "weights_sha256": _sha256(weights_path),
        "weight_dtype": np.dtype(dtype).name,
        "registry_entries": len(registry),
        "registry_shapes": {
            name: list(value.shape) for name, value in registry.items()
        },
        "layout_contract": {
            "activations": "NHWC",
            "filters": "HWIO",
            "outputs": "NHWC raw heads",
        },
        "experimental_only": True,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(
        json.dumps(
            export_checkpoint(
                arguments.checkpoint, arguments.weights, arguments.manifest
            ),
            sort_keys=True,
        )
    )
