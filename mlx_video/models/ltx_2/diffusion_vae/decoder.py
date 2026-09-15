"""LTX-2.5 diffusion video VAE decoder for MLX."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
from mlx import nn
from safetensors import safe_open

from mlx_video.models.ltx_2.diffusion_vae.layers import (
    LinearPixelShuffleUpsample,
    NABlock,
    NeighborhoodAttention3D,
    SwiGLU,
)


def timestep_embedding(timestep: mx.array, dimension: int = 256) -> mx.array:
    half = dimension // 2
    exponent = -math.log(10000) * mx.arange(half, dtype=mx.float32) / half
    angles = timestep[:, None].astype(mx.float32) * mx.exp(exponent)[None]
    return mx.concatenate([mx.cos(angles), mx.sin(angles)], axis=-1)


def patchify_pixels(x: mx.array, patch_size: int) -> mx.array:
    """BCTHW pixels to BTHW(C*r*q), matching the official c-r-q order."""
    batch, channels, frames, height, width = x.shape
    x = mx.reshape(
        x,
        (
            batch,
            channels,
            frames,
            height // patch_size,
            patch_size,
            width // patch_size,
            patch_size,
        ),
    )
    x = mx.transpose(x, (0, 2, 3, 5, 1, 6, 4))
    return mx.reshape(
        x,
        (
            batch,
            frames,
            height // patch_size,
            width // patch_size,
            channels * patch_size**2,
        ),
    )


def unpatchify_pixels(x: mx.array, patch_size: int, channels: int) -> mx.array:
    """BTHW(C*r*q) patches to BCTHW pixels."""
    batch, frames, height, width, _ = x.shape
    x = mx.reshape(
        x,
        (batch, frames, height, width, channels, patch_size, patch_size),
    )
    x = mx.transpose(x, (0, 4, 1, 2, 6, 3, 5))
    return mx.reshape(
        x,
        (
            batch,
            channels,
            frames,
            height * patch_size,
            width * patch_size,
        ),
    )


class TimestepEmbedder(nn.Module):
    def __init__(self, embedding_dim: int = 384) -> None:
        super().__init__()
        self.mlp = [
            nn.Linear(256, embedding_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim, bias=True),
        ]

    def __call__(self, timestep: mx.array) -> mx.array:
        hidden = timestep_embedding(timestep)
        for layer in self.mlp:
            hidden = layer(hidden)
        return hidden


class AdaLNZero(nn.Module):
    def __init__(self, dim: int, embedding_dim: int = 384) -> None:
        super().__init__()
        self.proj = nn.Linear(embedding_dim, 7 * dim, bias=True)

    def __call__(self, embedding: mx.array) -> tuple[mx.array, ...]:
        projected = self.proj(nn.silu(embedding))
        return tuple(
            part[:, None, None, None] for part in mx.split(projected, 7, axis=-1)
        )


class DiffusionNABlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        context_channels: int,
        head_dim: int = 64,
    ) -> None:
        super().__init__()
        self.context_proj = nn.Linear(context_channels, dim, bias=True)
        self.scale_shift_table = mx.zeros((7, dim))
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = NeighborhoodAttention3D(dim, kernel_size, head_dim)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.mlp = SwiGLU(dim, 4 * dim)

    def __call__(
        self,
        context: mx.array,
        x: mx.array,
        modulation: tuple[mx.array, ...],
    ) -> mx.array:
        values = tuple(
            modulation[index] + self.scale_shift_table[index][None, None, None, None]
            for index in range(7)
        )
        scale_msa, shift_msa, _, scale_mlp, shift_mlp, _, _ = values
        x = x + self.context_proj(context)
        attention_input = self.norm1(x) * (1 + scale_msa) + shift_msa
        x = x + self.attn(attention_input)
        mlp_input = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        return x + self.mlp(mlp_input)


class DiffusionVideoDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 3,
        patch_size: int = 4,
        head_dim: int = 64,
        stage_channels: tuple[int, ...] = (2048, 1024, 512, 512, 256),
        stage_depths: tuple[int, ...] = (4, 6, 4, 2, 8),
        stage_kernels: tuple[tuple[int, int, int], ...] = (
            (3, 7, 7),
            (3, 7, 7),
            (3, 5, 5),
            (3, 5, 5),
            (11, 11, 11),
        ),
        upsamples: tuple[tuple[tuple[int, int, int], int], ...] = (
            ((1, 2, 2), 2),
            ((2, 1, 1), 2),
            ((2, 2, 2), 1),
            ((2, 2, 2), 2),
        ),
        timestep_scale_multiplier: float = 1000.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.stage_channels = stage_channels
        self.stage_depths = stage_depths
        self.stage_kernels = stage_kernels
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.mean_of_means = mx.zeros((in_channels,))
        self.std_of_means = mx.ones((in_channels,))
        self.type_emb = mx.zeros((in_channels,))
        self.conv_in = nn.Linear(in_channels, stage_channels[0], bias=True)
        self.det_stages = []
        self.upsamples = []
        for index in range(4):
            self.det_stages.append(
                [
                    NABlock(stage_channels[index], stage_kernels[index], head_dim)
                    for _ in range(stage_depths[index])
                ]
            )
            stride, reduction = upsamples[index]
            self.upsamples.append(
                LinearPixelShuffleUpsample(stage_channels[index], stride, reduction)
            )

        stage5_channels = stage_channels[-1]
        pixel_channels = out_channels * patch_size**2
        self.conv_in_x_t = nn.Linear(pixel_channels, stage5_channels, bias=True)
        self.t_embedder = TimestepEmbedder()
        self.shared_adaln = AdaLNZero(stage5_channels)
        self.diff_blocks = [
            DiffusionNABlock(
                stage5_channels,
                stage_kernels[-1],
                stage5_channels,
                head_dim,
            )
            for _ in range(stage_depths[-1])
        ]
        self.norm_out = nn.RMSNorm(stage5_channels, eps=1e-6)
        self.conv_out = nn.Linear(stage5_channels, pixel_channels, bias=True)

    def _deterministic_context(self, latent: mx.array) -> mx.array:
        original_frames = 8 * (latent.shape[2] - 1) + 1
        latent = mx.concatenate(
            [latent, mx.repeat(latent[:, :, -1:], 2, axis=2)], axis=2
        )
        mean = self.mean_of_means[None, :, None, None, None]
        std = self.std_of_means[None, :, None, None, None]
        hidden = latent * std + mean
        hidden = self.conv_in(mx.transpose(hidden, (0, 2, 3, 4, 1)))
        for stage_index, blocks in enumerate(self.det_stages):
            for block in blocks:
                hidden = block(hidden)
            hidden = self.upsamples[stage_index](hidden)
        return hidden[:, : max(original_frames, self.stage_kernels[-1][0])]

    def __call__(self, latent: mx.array, *, seed: int = 0) -> mx.array:
        output_frames = 8 * (latent.shape[2] - 1) + 1
        context = self._deterministic_context(latent)
        batch, frames, height, width, _ = context.shape
        mx.random.seed(seed)
        pixels = mx.random.normal(
            (
                batch,
                self.out_channels,
                frames,
                height * self.patch_size,
                width * self.patch_size,
            ),
            dtype=context.dtype,
        )
        hidden = self.conv_in_x_t(patchify_pixels(pixels, self.patch_size))
        timestep = mx.ones((batch,), dtype=mx.float32)
        embedding = self.t_embedder(timestep * self.timestep_scale_multiplier)
        modulation = self.shared_adaln(embedding)
        for block in self.diff_blocks:
            hidden = block(context, hidden, modulation)
        output = self.conv_out(self.norm_out(hidden))
        pixels = unpatchify_pixels(output, self.patch_size, self.out_channels)
        return pixels[:, :, :output_frames]

    @staticmethod
    def sanitize(weights: dict[str, mx.array]) -> dict[str, mx.array]:
        sanitized: dict[str, mx.array] = {}
        for key, value in weights.items():
            if key == "per_channel_statistics.mean-of-means":
                sanitized["mean_of_means"] = value
                continue
            if key == "per_channel_statistics.std-of-means":
                sanitized["std_of_means"] = value
                continue
            if not key.startswith("decoder."):
                continue
            key = key.removeprefix("decoder.")
            if key.endswith(("attn.qkv.weight", "attn.qkv.bias")):
                prefix, suffix = key.rsplit("qkv", 1)
                query, key_value, value_value = mx.split(value, 3, axis=0)
                sanitized[f"{prefix}to_q{suffix}"] = query
                sanitized[f"{prefix}to_k{suffix}"] = key_value
                sanitized[f"{prefix}to_v{suffix}"] = value_value
            else:
                sanitized[key] = value
        return sanitized

    @classmethod
    def from_pretrained(cls, path: str | Path) -> DiffusionVideoDecoder:
        path = Path(path)
        with safe_open(path, framework="numpy") as file:
            metadata = file.metadata() or {}
        config = json.loads(metadata["config"])["vae"]["decoder"]
        model = cls(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            patch_size=config["patch_size"],
            head_dim=config["head_dim"],
            stage_channels=tuple(config["stage_channels"]),
            stage_depths=tuple(config["stage_depths"]),
            stage_kernels=tuple(tuple(item) for item in config["stage_kernels"]),
            upsamples=tuple(
                (tuple(stride), reduction) for stride, reduction in config["upsamples"]
            ),
            timestep_scale_multiplier=config["timestep_scale_multiplier"],
        )
        weights = cls.sanitize(mx.load(str(path)))
        model.load_weights(list(weights.items()), strict=True)
        mx.eval(model.parameters())
        model.eval()
        return model
