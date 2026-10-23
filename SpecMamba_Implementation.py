import math

import torch
from torch import nn
from torch.nn import functional as F


class RevIN(nn.Module):
    def __init__(self, n_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_features))
        self.bias = nn.Parameter(torch.zeros(n_features))

    def normalize(self, x):
        mean = x.mean(dim=1, keepdim=True).detach()
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        return (x - mean) / std * self.weight + self.bias, mean, std

    def denormalize(self, x, mean, std):
        return (x - self.bias) / (self.weight + 1e-10) * std + mean


class PatchEmbedding(nn.Module):
    def __init__(self, seq_len, patch_len, stride, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = math.floor((seq_len + stride - patch_len) / stride) + 1
        self.projection = nn.Linear(patch_len, d_model, bias=False)
        self.position = nn.Parameter(torch.randn(1, self.num_patches + 5, d_model))
        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        x = F.pad(x.transpose(1, 2), (0, self.stride), mode="replicate")
        x = x.unfold(-1, self.patch_len, self.stride).squeeze(1)
        x = self.projection(x)
        return self.dropout(x + self.position[:, : x.size(1)])


class FrequencyFilter(nn.Module):
    def __init__(self, num_patches, d_model, dropout, retained_bins=None):
        super().__init__()
        frequency_count = num_patches // 2 + 1
        if retained_bins is not None and not 1 <= retained_bins <= frequency_count:
            raise ValueError(f"filter_bins must be in [1, {frequency_count}]")
        self.retained_bins = retained_bins
        self.real = nn.Parameter(torch.randn(frequency_count, d_model) * 0.02)
        self.imag = nn.Parameter(torch.randn(frequency_count, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        spectrum = torch.fft.rfft(x, dim=1)
        weights = torch.complex(self.real, self.imag)
        if self.retained_bins is not None:
            mask = torch.arange(weights.size(0), device=x.device) < self.retained_bins
            weights = weights * mask[:, None]
        return self.dropout(torch.fft.irfft(spectrum * weights, n=x.size(1), dim=1))


class SpecSSM(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config["fusion"] not in ("gated", "fixed_add", "concat_mlp"):
            raise ValueError("Unknown fusion variant")
        if not config["use_spec"] and config["fusion"] != "gated":
            raise ValueError("Fusion variants require the spectral branch")
        try:
            from mamba_ssm import Mamba
        except ImportError as exc:
            raise RuntimeError("mamba_ssm is required for Spec-SSM") from exc
        self.config = config
        self.revin = RevIN(config["n_features"])
        self.patch = PatchEmbedding(config["seq_len"], config["patch_len"], config["stride"], config["d_model"])
        self.trend = nn.Linear(config["seq_len"], config["horizon"]) if config["use_trend"] else None
        if self.trend is not None:
            nn.init.normal_(self.trend.weight, 0, 0.01)
        self.mamba_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(config["d_model"]),
                Mamba(d_model=config["d_model"], d_state=config["d_state"], d_conv=config["d_conv"], expand=config["expand"]),
                nn.Dropout(config["dropout"]),
            )
            for _ in range(config["e_layers"])
        ])
        self.spectral = FrequencyFilter(self.patch.num_patches, config["d_model"], config["dropout"], config["filter_bins"]) if config["use_spec"] else None
        self.alpha = nn.Parameter(torch.tensor(float(config["alpha_init"]))) if config["use_spec"] and config["fusion"] == "gated" else None
        self.fusion_mlp = nn.Sequential(nn.Linear(2 * config["d_model"], config["d_model"]), nn.GELU(), nn.Linear(config["d_model"], config["d_model"])) if config["use_spec"] and config["fusion"] == "concat_mlp" else None
        self.head = nn.Linear(self.patch.num_patches * config["d_model"], config["horizon"])
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        batch, length, channels = x.shape
        if length != self.config["seq_len"] or channels != self.config["n_features"]:
            raise ValueError(f"Expected [batch, {self.config['seq_len']}, {self.config['n_features']}]")
        x, mean, std = self.revin.normalize(x)
        channel_input = x.transpose(1, 2).reshape(batch * channels, length)
        trend = self.trend(channel_input) if self.trend is not None else 0
        shared = self.patch(channel_input.unsqueeze(-1))
        temporal = shared
        for layer in self.mamba_layers:
            temporal = temporal + layer(temporal)
        if self.spectral is not None:
            frequency = self.spectral(shared)
            if self.config["fusion"] == "gated":
                temporal = temporal + self.alpha * frequency
            elif self.config["fusion"] == "fixed_add":
                temporal = temporal + frequency
            else:
                temporal = self.fusion_mlp(torch.cat((temporal, frequency), dim=-1))
        seasonal = self.dropout(self.head(temporal.reshape(batch * channels, -1)))
        prediction = (trend + seasonal).reshape(batch, channels, self.config["horizon"]).transpose(1, 2)
        return self.revin.denormalize(prediction, mean, std)
