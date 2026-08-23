from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class ModelSpec:
    family: str
    input_dim: int
    feature_shape: tuple[int, ...]
    num_classes: int
    hidden_dims: tuple[int, int]
    channel_dims: tuple[int, int]
    cnn_hidden_dim: int
    cnn_variant: str
    conv_feature_shape: tuple[int, int]
    classifier_weight_name: str
    classifier_bias_name: str


def _conv_feature_shape(feature_shape: tuple[int, ...]) -> tuple[int, int]:
    if len(feature_shape) != 3:
        raise ValueError(f'CNN feature shape must be (C, H, W), got {feature_shape}')
    _, height, width = feature_shape
    height = max(1, height // 4)
    width = max(1, width // 4)
    return int(height), int(width)


def build_model_spec(
    input_dim: int,
    num_classes: int,
    feature_shape: tuple[int, ...],
    hidden_dims: tuple[int, int] = (128, 64),
    model_family: str = 'auto',
    channel_dims: tuple[int, int] = (16, 32),
    cnn_hidden_dim: int = 64,
    cnn_variant: str = 'basic',
) -> ModelSpec:
    resolved_family = model_family
    if resolved_family == 'auto':
        resolved_family = 'cnn' if len(feature_shape) == 3 and max(feature_shape[-2:]) >= 16 else 'mlp'
    if resolved_family not in {'mlp', 'cnn'}:
        raise ValueError(f'Unsupported model family: {resolved_family}')
    if cnn_variant not in {'basic', 'residual'}:
        raise ValueError(f'Unsupported CNN variant: {cnn_variant}')
    if resolved_family == 'mlp':
        return ModelSpec(
            family='mlp',
            input_dim=input_dim,
            feature_shape=feature_shape,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            channel_dims=channel_dims,
            cnn_hidden_dim=cnn_hidden_dim,
            cnn_variant='basic',
            conv_feature_shape=(1, 1),
            classifier_weight_name='fc3.weight',
            classifier_bias_name='fc3.bias',
        )
    return ModelSpec(
        family='cnn',
        input_dim=input_dim,
        feature_shape=feature_shape,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        channel_dims=channel_dims,
        cnn_hidden_dim=cnn_hidden_dim,
        cnn_variant=cnn_variant,
        conv_feature_shape=_conv_feature_shape(feature_shape),
        classifier_weight_name='fc2.weight',
        classifier_bias_name='fc2.bias',
    )


class FederatedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: tuple[int, int] = (128, 64),
    ) -> None:
        super().__init__()
        hidden_one, hidden_two = hidden_dims
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.fc1 = nn.Linear(input_dim, hidden_one)
        self.fc2 = nn.Linear(hidden_one, hidden_two)
        self.fc3 = nn.Linear(hidden_two, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class FederatedCNN(nn.Module):
    def __init__(
        self,
        feature_shape: tuple[int, ...],
        num_classes: int,
        channel_dims: tuple[int, int] = (16, 32),
        hidden_dim: int = 64,
        variant: str = 'basic',
    ) -> None:
        super().__init__()
        if len(feature_shape) != 3:
            raise ValueError(f'CNN feature shape must be (C, H, W), got {feature_shape}')
        in_channels = int(feature_shape[0])
        conv_one, conv_two = channel_dims
        pooled_h, pooled_w = _conv_feature_shape(feature_shape)
        self.conv_feature_shape = (pooled_h, pooled_w)
        self.conv1 = nn.Conv2d(in_channels, conv_one, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv_one, conv_two, kernel_size=3, padding=1)
        self.variant = variant
        if variant not in {'basic', 'residual'}:
            raise ValueError(f'Unsupported CNN variant: {variant}')
        if variant == 'residual':
            self.norm1 = nn.GroupNorm(num_groups=min(8, conv_one), num_channels=conv_one, affine=False)
            self.norm2 = nn.GroupNorm(num_groups=min(8, conv_two), num_channels=conv_two, affine=False)
            self.conv1_residual = nn.Conv2d(conv_one, conv_one, kernel_size=3, padding=1)
            self.conv2_residual = nn.Conv2d(conv_two, conv_two, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(conv_two * pooled_h * pooled_w, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        if self.variant == 'residual':
            x = self.norm1(x)
            x = self.relu(x)
            x = self.relu(self.conv1_residual(x) + x)
        else:
            x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        if self.variant == 'residual':
            x = self.norm2(x)
            x = self.relu(x)
            x = self.relu(self.conv2_residual(x) + x)
        else:
            x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class DigitsMLP(FederatedMLP):
    def __init__(self, hidden_dims: tuple[int, int] = (128, 64)) -> None:
        super().__init__(input_dim=64, num_classes=10, hidden_dims=hidden_dims)


def build_model(model_spec: ModelSpec) -> nn.Module:
    if model_spec.family == 'mlp':
        return FederatedMLP(
            input_dim=model_spec.input_dim,
            num_classes=model_spec.num_classes,
            hidden_dims=model_spec.hidden_dims,
        )
    return FederatedCNN(
        feature_shape=model_spec.feature_shape,
        num_classes=model_spec.num_classes,
        channel_dims=model_spec.channel_dims,
        hidden_dim=model_spec.cnn_hidden_dim,
        variant=model_spec.cnn_variant,
    )


def clone_state_dict(state_dict: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((name, tensor.detach().clone()) for name, tensor in state_dict.items())


def subtract_state_dicts(
    newer: OrderedDict[str, torch.Tensor],
    older: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((name, newer[name] - older[name]) for name in newer)


def add_delta_in_place(
    state_dict: OrderedDict[str, torch.Tensor],
    delta: OrderedDict[str, torch.Tensor],
    scale: float = 1.0,
) -> None:
    for name in state_dict:
        state_dict[name].add_(delta[name], alpha=scale)


def average_deltas(
    deltas: Iterable[OrderedDict[str, torch.Tensor]],
    weights: list[float],
) -> OrderedDict[str, torch.Tensor]:
    deltas = list(deltas)
    first = deltas[0]
    output = OrderedDict((name, torch.zeros_like(first[name])) for name in first)
    total = sum(weights)
    if total == 0:
        raise ValueError('Aggregation weights sum to zero.')
    for delta, weight in zip(deltas, weights):
        norm_weight = weight / total
        for name in output:
            output[name].add_(delta[name], alpha=norm_weight)
    return output


def flatten_state_dict(state_dict: OrderedDict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensor.reshape(-1).detach().cpu() for tensor in state_dict.values()])
