from __future__ import annotations

import torch
from torch import nn

from .encoding import ACTION_SIZE, BOARD_CHANNELS


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ChessStudent(nn.Module):
    def __init__(self, channels: int = 64, depth: int = 4, depthwise: bool = False) -> None:
        super().__init__()
        block_type = DepthwiseSeparableBlock if depthwise else ConvBlock
        layers: list[nn.Module] = [ConvBlock(BOARD_CHANNELS, channels)]
        layers.extend(block_type(channels, channels) for _ in range(max(0, depth - 1)))
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 64 * 5, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(board)
        return self.policy_head(features), self.value_head(features)


def build_model(name: str, channels: int = 64, depth: int = 4) -> ChessStudent:
    if name == "small_cnn":
        return ChessStudent(channels=channels, depth=depth, depthwise=False)
    if name == "depthwise_cnn":
        return ChessStudent(channels=channels, depth=depth, depthwise=True)
    raise ValueError(f"Unknown model name: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
