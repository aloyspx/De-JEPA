from collections.abc import Sequence

import torch
from torch import nn


class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.layers = nn.Sequential(
            nn.BatchNorm3d(channels),
            nn.ReLU(),
            nn.Conv3d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm3d(channels),
            nn.ReLU(),
            nn.Conv3d(channels, channels, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class ResidualStage(nn.Module):
    def __init__(self, channels: int, blocks: int = 2) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*(ResidualBlock3D(channels) for _ in range(blocks)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class FPN3D(nn.Module):
    def __init__(
        self, in_channels: int = 1, base_channels: int = 16, scales: int = 6
    ) -> None:
        super().__init__()
        channels = [base_channels * 2**i for i in range(scales)]
        self.first = nn.Conv3d(in_channels, channels[0], kernel_size=3, padding=1)
        self.encoder = nn.ModuleList(ResidualStage(c) for c in channels[:-1])
        self.down = nn.ModuleList(
            nn.Sequential(
                nn.MaxPool3d(2, ceil_mode=True), nn.Conv3d(a, b, kernel_size=1)
            )
            for a, b in zip(channels[:-1], channels[1:])
        )
        self.bottom = ResidualStage(channels[-1])
        self.up = nn.ModuleList(
            nn.Sequential(
                nn.Conv3d(b, a, kernel_size=1),
                nn.Upsample(scale_factor=2, mode="nearest"),
            )
            for a, b in reversed(list(zip(channels[:-1], channels[1:])))
        )
        self.skip = nn.ModuleList(
            nn.Conv3d(c, c, kernel_size=1) for c in reversed(channels[:-1])
        )
        self.decoder = nn.ModuleList(ResidualStage(c) for c in reversed(channels[:-1]))
        self.channels = channels
        self.scales = scales

    @property
    def embedding_size(self) -> int:
        return sum(self.channels)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.first(x)
        pyramid = []
        for stage, down in zip(self.encoder, self.down):
            x = stage(x)
            pyramid.append(x)
            x = down(x)
        x = self.bottom(x)
        output = [x]
        for up, skip, stage, lateral in zip(
            self.up, self.skip, self.decoder, reversed(pyramid)
        ):
            x = up(x)
            x = x[(..., *map(slice, lateral.shape[-3:]))]
            x = stage(x + skip(lateral))
            output.insert(0, x)
        return output


class Projector(nn.Sequential):
    def __init__(
        self, input_dim: int, output_dim: int = 128, hidden_dim: int | None = None
    ) -> None:
        hidden_dim = input_dim if hidden_dim is None else hidden_dim
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )


def select_from_pyramid(
    pyramid: Sequence[torch.Tensor], coordinates: torch.Tensor
) -> torch.Tensor:
    batch, points, _ = coordinates.shape
    batch_index = (
        torch.arange(batch, device=coordinates.device)
        .view(batch, 1)
        .expand(batch, points)
    )
    selected = []
    for scale, feature_map in enumerate(pyramid):
        scaled = coordinates // 2**scale
        depth, height, width = scaled.unbind(-1)
        selected.append(feature_map[batch_index, :, depth, height, width])
    return torch.cat(selected, dim=-1)


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.moveaxis(1, -1)).moveaxis(-1, 1)


class LinearHead(nn.Module):
    def __init__(
        self, channels: Sequence[int], classes: int, normalize: bool = False
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Sequential(
                ChannelLayerNorm(c) if normalize else nn.Identity(),
                nn.Conv3d(c, classes, 1, bias=i == 0),
            )
            for i, c in enumerate(channels)
        )

    def forward(self, pyramid: Sequence[torch.Tensor]) -> torch.Tensor:
        logits = [layer(x) for layer, x in zip(self.layers, pyramid)]
        x = logits[-1]
        for lateral in reversed(logits[:-1]):
            x = (
                nn.functional.interpolate(x, size=lateral.shape[-3:], mode="nearest")
                + lateral
            )
        return x


class NonlinearHead(nn.Module):
    def __init__(self, channels: Sequence[int], classes: int) -> None:
        super().__init__()
        self.bottom = ResidualBlock3D(channels[-1], kernel_size=1)
        pairs = list(zip(channels[:-1], channels[1:]))[::-1]
        self.up = nn.ModuleList(
            nn.Conv3d(source, target, 1) for target, source in pairs
        )
        self.skip = nn.ModuleList(nn.Conv3d(target, target, 1) for target, _ in pairs)
        self.refine = nn.ModuleList(
            ResidualBlock3D(target, kernel_size=1) for target, _ in pairs
        )
        self.output = nn.Conv3d(channels[0], classes, 1)

    def forward(self, pyramid: Sequence[torch.Tensor]) -> torch.Tensor:
        x = self.bottom(pyramid[-1])
        for up, skip, refine, lateral in zip(
            self.up, self.skip, self.refine, reversed(pyramid[:-1])
        ):
            x = nn.functional.interpolate(
                up(x), size=lateral.shape[-3:], mode="nearest"
            )
            x = refine(x + skip(lateral))
        return self.output(x)
