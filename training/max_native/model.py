"""Trainable PyTorch twin of the benchmarked MAX-native detector graph."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

BLOCKS = (
    (16, 16, 16, 1),
    (16, 24, 64, 2),
    (24, 24, 72, 1),
    (24, 40, 72, 2),
    (40, 40, 120, 1),
    (40, 40, 120, 1),
    (40, 80, 240, 2),
    (80, 80, 200, 1),
    (80, 80, 184, 1),
    (80, 80, 184, 1),
    (80, 112, 480, 1),
    (112, 112, 672, 1),
    (112, 160, 672, 2),
    (160, 160, 960, 1),
    (160, 160, 960, 1),
)


@dataclass(frozen=True)
class ArchitectureSpec:
    version: int = 4
    layout: str = "NHWC"
    dtype: str = "float16"
    classes: tuple[str, ...] = ("pear", "apple", "banana")
    stem_channels: int = 16
    blocks: tuple[tuple[int, int, int, int], ...] = BLOCKS
    head_strides: tuple[int, ...] = (8, 16, 32)
    head_channels: int = 8
    feature_pyramid_channels: int = 64
    box_encoding: str = "ltrb_exp_grid_center"
    postprocessing_in_graph: bool = False

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "layout": self.layout,
            "dtype": self.dtype,
            "classes": list(self.classes),
            "stem_channels": self.stem_channels,
            "blocks": self.blocks,
            "head_strides": list(self.head_strides),
            "head_channels": self.head_channels,
            "feature_pyramid_channels": self.feature_pyramid_channels,
            "box_encoding": self.box_encoding,
            "postprocessing_in_graph": self.postprocessing_in_graph,
        }

    def max_output_shapes(
        self, resolution: int, batch_size: int = 1
    ) -> tuple[tuple[int, int, int, int], ...]:
        if resolution <= 0 or resolution % max(self.head_strides) != 0:
            raise ValueError(
                f"resolution must be positive and divisible by {max(self.head_strides)}"
            )
        return tuple(
            (
                batch_size,
                resolution // stride,
                resolution // stride,
                self.head_channels,
            )
            for stride in self.head_strides
        )


DEFAULT_SPEC = ArchitectureSpec()
ARCHITECTURE_SHA256 = hashlib.sha256(
    json.dumps(DEFAULT_SPEC.manifest(), sort_keys=True).encode("utf-8")
).hexdigest()


class ConvNorm(nn.Module):
    """Convolution plus foldable batch normalization."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups,
            bias=False,
        )
        # The randomly initialized pyramid and detection heads need their
        # deployment statistics to track training quickly. A 0.1 update rate
        # avoids the train/eval divergence observed with the backbone-oriented
        # 0.03 setting while remaining exactly foldable for MAX export.
        self.norm = nn.BatchNorm2d(output_channels, eps=1e-3, momentum=0.1)

    def forward(self, value: Tensor) -> Tensor:
        return self.norm(self.conv(value))


class InvertedResidual(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        expanded_channels: int,
        stride: int,
    ) -> None:
        super().__init__()
        self.expand = (
            None
            if expanded_channels == input_channels
            else ConvNorm(input_channels, expanded_channels, 1)
        )
        self.depthwise = ConvNorm(
            expanded_channels,
            expanded_channels,
            3,
            stride=stride,
            groups=expanded_channels,
        )
        self.project = ConvNorm(expanded_channels, output_channels, 1)
        self.use_residual = stride == 1 and input_channels == output_channels
        self.activation = nn.ReLU(inplace=False)

    def forward(self, value: Tensor) -> Tensor:
        residual = value
        if self.expand is not None:
            value = self.activation(self.expand(value))
        value = self.activation(self.depthwise(value))
        value = self.project(value)
        if self.use_residual:
            return value + residual
        return self.activation(value)


class DepthwiseSeparableBlock(nn.Module):
    """A MAX-friendly spatial refinement block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = ConvNorm(channels, channels, 3, groups=channels)
        self.pointwise = ConvNorm(channels, channels, 1)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, value: Tensor) -> Tensor:
        value = self.activation(self.depthwise(value))
        return self.activation(self.pointwise(value))


class DetectionHead(nn.Module):
    """A small spatial head ending in the stable raw detection interface."""

    def __init__(self, channels: int, output_channels: int) -> None:
        super().__init__()
        self.depthwise = ConvNorm(channels, channels, 3, groups=channels)
        self.pointwise = ConvNorm(channels, channels, 1)
        self.output = nn.Conv2d(channels, output_channels, 1, bias=True)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, value: Tensor) -> Tensor:
        value = self.activation(self.depthwise(value))
        value = self.activation(self.pointwise(value))
        return self.output(value)


class TrainableFruitDetector(nn.Module):
    """The training adapter for the exact MAX graph architecture.

    PyTorch uses NCHW internally. ``named_export_layers`` exposes every
    convolution under the MAX weight-registry name so export can fold batch
    normalization and transpose filters into HWIO without callers knowing the
    model's internal module layout.
    """

    def __init__(self, spec: ArchitectureSpec = DEFAULT_SPEC) -> None:
        super().__init__()
        if spec.head_channels != 5 + len(spec.classes):
            raise ValueError(
                "head_channels must contain four boxes, objectness, and one logit per class"
            )
        self.spec = spec
        self.stem = ConvNorm(3, spec.stem_channels, 3, stride=2)
        self.blocks = nn.ModuleList(
            InvertedResidual(*block) for block in spec.blocks
        )
        feature_channels: dict[int, int] = {}
        total_stride = 2
        for _, output_channels, _, stride in spec.blocks:
            total_stride *= stride
            if total_stride in spec.head_strides:
                feature_channels[total_stride] = output_channels
        if set(feature_channels) != set(spec.head_strides):
            raise ValueError("backbone does not produce every configured detection stride")
        pyramid_channels = spec.feature_pyramid_channels
        self.laterals = nn.ModuleDict(
            {
                str(stride): ConvNorm(feature_channels[stride], pyramid_channels, 1)
                for stride in spec.head_strides
            }
        )
        self.refinements = nn.ModuleDict(
            {
                str(stride): DepthwiseSeparableBlock(pyramid_channels)
                for stride in spec.head_strides
            }
        )
        self.heads = nn.ModuleDict(
            {
                str(stride): DetectionHead(pyramid_channels, spec.head_channels)
                for stride in spec.head_strides
            }
        )
        for head in self.heads.values():
            nn.init.normal_(head.output.weight, mean=0.0, std=0.01)
            nn.init.zeros_(head.output.bias)
            with torch.no_grad():
                # Objectness begins with a conservative ~1% prior. Class
                # logits remain neutral so their score is not suppressed a
                # second time before the classifier has learned anything.
                head.output.bias[4].fill_(-4.59511985013459)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, value: Tensor) -> tuple[Tensor, ...]:
        value = self.activation(self.stem(value))
        total_stride = 2
        features: dict[int, Tensor] = {}
        for block, (_, _, _, stride) in zip(
            self.blocks, self.spec.blocks, strict=True
        ):
            value = block(value)
            total_stride *= stride
            if total_stride in self.spec.head_strides:
                features[total_stride] = value
        pyramid: dict[int, Tensor] = {}
        previous: Tensor | None = None
        for stride in reversed(self.spec.head_strides):
            lateral = self.laterals[str(stride)](features[stride])
            if previous is not None:
                lateral = lateral + F.interpolate(
                    previous, size=lateral.shape[-2:], mode="nearest"
                )
            pyramid[stride] = self.refinements[str(stride)](
                self.activation(lateral)
            )
            previous = pyramid[stride]
        return tuple(
            self.heads[str(stride)](pyramid[stride])
            for stride in self.spec.head_strides
        )

    def named_export_layers(self) -> Iterator[tuple[str, nn.Conv2d]]:
        yield "stem", self.stem.conv
        for index, block in enumerate(self.blocks):
            if block.expand is not None:
                yield f"blocks.{index}.expand", block.expand.conv
            yield f"blocks.{index}.depthwise", block.depthwise.conv
            yield f"blocks.{index}.project", block.project.conv
        for stride in self.spec.head_strides:
            yield f"laterals.stride_{stride}", self.laterals[str(stride)].conv
        for stride in self.spec.head_strides:
            refinement = self.refinements[str(stride)]
            yield f"refinements.stride_{stride}.depthwise", refinement.depthwise.conv
            yield f"refinements.stride_{stride}.pointwise", refinement.pointwise.conv
        for stride in self.spec.head_strides:
            head = self.heads[str(stride)]
            yield f"heads.stride_{stride}.depthwise", head.depthwise.conv
            yield f"heads.stride_{stride}.pointwise", head.pointwise.conv
            yield f"heads.stride_{stride}.output", head.output

    def named_foldable_layers(self) -> Iterator[tuple[str, ConvNorm]]:
        yield "stem", self.stem
        for index, block in enumerate(self.blocks):
            if block.expand is not None:
                yield f"blocks.{index}.expand", block.expand
            yield f"blocks.{index}.depthwise", block.depthwise
            yield f"blocks.{index}.project", block.project
        for stride in self.spec.head_strides:
            yield f"laterals.stride_{stride}", self.laterals[str(stride)]
        for stride in self.spec.head_strides:
            refinement = self.refinements[str(stride)]
            yield f"refinements.stride_{stride}.depthwise", refinement.depthwise
            yield f"refinements.stride_{stride}.pointwise", refinement.pointwise
        for stride in self.spec.head_strides:
            head = self.heads[str(stride)]
            yield f"heads.stride_{stride}.depthwise", head.depthwise
            yield f"heads.stride_{stride}.pointwise", head.pointwise
