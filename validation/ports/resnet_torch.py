"""Native PyTorch port of Flax ImageNet ResNet V1.5.

Derived from google/flax examples/imagenet/models.py at the revision recorded in
validation/originals/manifest.json. Original copyright: 2024 The Flax Authors.
Licensed under Apache-2.0; see ../originals/flax_imagenet/LICENSE.

Supports dense convolutions, basic/bottleneck blocks, NHWC inputs and explicit
Flax parameter/state conversion. ConvLocal and cross-device BatchNorm are outside
this bounded port's scope. No JAX runtime is required for model execution.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def same_padding(height: int, width: int, kernel: int, stride: int) -> tuple[int, ...]:
    """JAX SAME padding places an odd extra element on the high side."""
    ph = max(((height + stride - 1) // stride - 1) * stride + kernel - height, 0)
    pw = max(((width + stride - 1) // stride - 1) * stride + kernel - width, 0)
    return pw // 2, pw - pw // 2, ph // 2, ph - ph // 2


class SameConv(nn.Conv2d):
    def __init__(self, inputs: int, outputs: int, kernel: int, stride: int = 1):
        super().__init__(inputs, outputs, kernel, stride=stride, padding=0, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        padding = same_padding(*inputs.shape[-2:], self.kernel_size[0], self.stride[0])
        return F.conv2d(F.pad(inputs, padding), self.weight, stride=self.stride)


class FlaxBatchNorm(nn.Module):
    """Population-variance moving state, matching the selected Linen source."""
    def __init__(self, channels: int, *, zero_scale: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels) if zero_scale else torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer('running_mean', torch.zeros(channels))
        self.register_buffer('running_var', torch.ones(channels))

    def forward(self, inputs: Tensor) -> Tensor:
        values = inputs.float() if inputs.dtype in (torch.float16, torch.bfloat16) else inputs
        if self.training:
            mean = values.mean((0, 2, 3))
            variance = (values.square().mean((0, 2, 3)) - mean.square()).clamp_min(0)
            with torch.no_grad():
                self.running_mean.mul_(0.9).add_(mean.detach(), alpha=0.1)
                self.running_var.mul_(0.9).add_(variance.detach(), alpha=0.1)
        else:
            mean, variance = self.running_mean, self.running_var
        scale = torch.rsqrt(variance + 1e-5) * self.weight
        output = (values - mean[None, :, None, None]) * scale[None, :, None, None]
        return (output + self.bias[None, :, None, None]).to(inputs.dtype)


class ResidualBlock(nn.Module):
    def __init__(self, inputs: int, filters: int, stride: int, bottleneck: bool):
        super().__init__()
        self.bottleneck = bottleneck
        outputs = filters * 4 if bottleneck else filters
        self.Conv_0 = SameConv(inputs, filters, 1 if bottleneck else 3, 1 if bottleneck else stride)
        self.BatchNorm_0 = FlaxBatchNorm(filters)
        self.Conv_1 = SameConv(filters, filters if bottleneck else outputs, 3, stride if bottleneck else 1)
        self.BatchNorm_1 = FlaxBatchNorm(filters, zero_scale=not bottleneck)
        if bottleneck:
            self.Conv_2 = SameConv(filters, outputs, 1)
            self.BatchNorm_2 = FlaxBatchNorm(outputs, zero_scale=True)
        if inputs != outputs or stride != 1:
            self.conv_proj = SameConv(inputs, outputs, 1, stride)
            self.norm_proj = FlaxBatchNorm(outputs)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = inputs
        values = F.relu(self.BatchNorm_0(self.Conv_0(inputs)))
        values = self.BatchNorm_1(self.Conv_1(values))
        if self.bottleneck:
            values = self.BatchNorm_2(self.Conv_2(F.relu(values)))
        if hasattr(self, 'conv_proj'):
            residual = self.norm_proj(self.conv_proj(residual))
        return F.relu(residual + values)


class ResNet(nn.Module):
    def __init__(self, stage_sizes: Sequence[int] = (2, 2, 2, 2), *, num_classes: int = 1000,
                 num_filters: int = 64, bottleneck: bool = False):
        super().__init__()
        if not stage_sizes or any(size < 1 for size in stage_sizes) or min(num_classes, num_filters) < 1:
            raise ValueError('Stage sizes, class count and filter count must be positive')
        self.conv_init = nn.Conv2d(3, num_filters, 7, stride=2, padding=3, bias=False)
        self.bn_init = FlaxBatchNorm(num_filters)
        names = []
        channels = num_filters
        for stage, count in enumerate(stage_sizes):
            for index in range(count):
                name = ('BottleneckResNetBlock_' if bottleneck else 'ResNetBlock_') + str(len(names))
                filters = num_filters * 2**stage
                self.add_module(name, ResidualBlock(channels, filters, 2 if stage and index == 0 else 1, bottleneck))
                channels = filters * (4 if bottleneck else 1)
                names.append(name)
        self.block_names = tuple(names)
        self.Dense_0 = nn.Linear(channels, num_classes)
        # Distribution-equivalent LeCun initialization; cross-framework seeds are not draw-equivalent.
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                fan_in = module.weight[0].numel()
                scale = math.sqrt(1.0 / fan_in) / 0.8796256610342398
                nn.init.trunc_normal_(module.weight, std=scale, a=-2 * scale, b=2 * scale)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[-1] != 3:
            raise ValueError('Expected NHWC inputs with three channels')
        values = inputs.permute(0, 3, 1, 2)
        values = F.relu(self.bn_init(self.conv_init(values)))
        values = F.max_pool2d(F.pad(values, same_padding(*values.shape[-2:], 3, 2), value=-math.inf), 3, 2)
        for name in self.block_names:
            values = getattr(self, name)(values)
        return self.Dense_0(values.mean((2, 3)))


def flatten_tree(tree: Mapping, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], object]:
    output = {}
    for key, value in tree.items():
        path = prefix + (key,)
        if isinstance(value, Mapping):
            output.update(flatten_tree(value, path))
        else:
            output[path] = value
    return output


def torch_key(path: tuple[str, ...]) -> str:
    replacement = {'kernel': 'weight', 'scale': 'weight', 'mean': 'running_mean', 'var': 'running_var'}
    return '.'.join((*path[:-1], replacement.get(path[-1], path[-1])))


def canonical_to_torch(path: tuple[str, ...], value: object) -> np.ndarray:
    array = np.asarray(value)
    if path[-1] == 'kernel':
        array = array.transpose(3, 2, 0, 1) if array.ndim == 4 else array.T
    return np.array(array, copy=True, order='C')


def load_flax(model: ResNet, variables: Mapping) -> list[dict]:
    if set(variables) != {'params', 'batch_stats'}:
        raise ValueError('Expected exactly params and batch_stats collections')
    state = {}
    manifest = []
    for collection, tree in variables.items():
        for path, value in flatten_tree(tree).items():
            name = torch_key(path)
            if name in state:
                raise ValueError(f'Duplicate target mapping: {name}')
            array = canonical_to_torch(path, value)
            state[name] = torch.from_numpy(array)
            manifest.append({'source': collection + '/' + '/'.join(path), 'target': name,
                             'source_shape': list(np.shape(value)), 'target_shape': list(array.shape),
                             'transform': 'HWIO-to-OIHW' if np.ndim(value) == 4 else
                                          'IO-to-OI' if path[-1] == 'kernel' else 'identity'})
    expected = model.state_dict()
    if set(expected) != set(state):
        raise ValueError(f'Mapping mismatch: missing={set(expected)-set(state)}, extra={set(state)-set(expected)}')
    for name, value in state.items():
        if value.shape != expected[name].shape or value.dtype != expected[name].dtype:
            raise ValueError(f'Shape/dtype mismatch for {name}')
    model.load_state_dict(state, strict=True)
    return manifest
