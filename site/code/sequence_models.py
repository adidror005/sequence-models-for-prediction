"""Sequence forecasting models used by the Medium series.

Every model accepts a float tensor shaped [batch, lookback, n_features]
and returns a direct multi-horizon forecast shaped [batch, horizon].
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class LinearForecaster(nn.Module):
    def __init__(self, lookback: int, n_features: int, horizon: int) -> None:
        super().__init__()
        self.fc = nn.Linear(lookback * n_features, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(start_dim=1))


class MLPForecaster(nn.Module):
    def __init__(
        self,
        lookback: int,
        n_features: int,
        horizon: int,
        hidden: int = 256,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(lookback * n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        n_features: int,
        horizon: int,
        hidden: int = 64,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states, _ = self.rnn(x)
        final_state = states[:, -1, :]
        return self.head(self.dropout(final_state))


class GRUForecaster(nn.Module):
    def __init__(
        self,
        n_features: int,
        horizon: int,
        hidden: int = 64,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=n_features,
            hidden_size=hidden,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states, _ = self.rnn(x)
        final_state = states[:, -1, :]
        return self.head(self.dropout(final_state))


class CNN1DForecaster(nn.Module):
    def __init__(
        self,
        n_features: int,
        horizon: int,
        channels: int = 64,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_features, channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(channels, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_first = x.transpose(1, 2)
        summary = self.features(channel_first).squeeze(-1)
        return self.head(summary)


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.left_padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.left_padding:
            y = y[:, :, :-self.left_padding]
        return y


class TCNForecaster(nn.Module):
    def __init__(
        self,
        n_features: int,
        horizon: int,
        channels: int = 64,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 4, 8),
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = n_features

        for index, dilation in enumerate(dilations):
            layers.extend(
                [
                    CausalConv1d(
                        in_channels,
                        channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                    ),
                    nn.ReLU(),
                ]
            )
            if index < len(dilations) - 1:
                layers.append(nn.Dropout(dropout))
            in_channels = channels

        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(channels, horizon)
        self.receptive_field = 1 + (kernel_size - 1) * sum(dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_first = x.transpose(1, 2)
        states = self.features(channel_first)
        return self.head(states[:, :, -1])


class TransformerForecaster(nn.Module):
    def __init__(
        self,
        lookback: int,
        n_features: int,
        horizon: int,
        d_model: int = 64,
        nhead: int = 4,
        layers: int = 2,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.input_projection = nn.Linear(n_features, d_model)
        self.position = nn.Parameter(torch.zeros(1, lookback, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.position.size(1):
            raise ValueError("input sequence exceeds configured lookback")
        z = self.input_projection(x)
        z = z + self.position[:, : x.size(1), :]
        z = self.encoder(z)
        final_token = self.norm(z[:, -1, :])
        return self.head(final_token)


class PatchTransformerForecaster(nn.Module):
    def __init__(
        self,
        lookback: int,
        n_features: int,
        horizon: int,
        patch_len: int = 12,
        stride: int = 12,
        d_model: int = 64,
        nhead: int = 4,
        layers: int = 2,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if lookback < patch_len:
            raise ValueError("lookback must be at least patch_len")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.patch_len = patch_len
        self.stride = stride
        n_patches = 1 + (lookback - patch_len) // stride

        self.patch_projection = nn.Linear(patch_len * n_features, d_model)
        self.position = nn.Parameter(torch.zeros(1, n_patches, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 1, 3, 2).flatten(start_dim=2)
        if patches.size(1) > self.position.size(1):
            raise ValueError("input creates more patches than configured")

        z = self.patch_projection(patches)
        z = z + self.position[:, : z.size(1), :]
        z = self.encoder(z)
        pooled = self.norm(z.mean(dim=1))
        return self.head(pooled)


class NBEATSBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        horizon: int,
        hidden: int = 256,
        depth: int = 3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = input_dim

        for _ in range(depth):
            layers.extend([nn.Linear(width, hidden), nn.ReLU()])
            width = hidden

        self.body = nn.Sequential(*layers)
        self.backcast_head = nn.Linear(hidden, input_dim)
        self.forecast_head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(x)
        return self.backcast_head(hidden), self.forecast_head(hidden)


class NBEATSForecaster(nn.Module):
    def __init__(
        self,
        lookback: int,
        n_features: int,
        horizon: int,
        n_blocks: int = 3,
        hidden: int = 256,
        depth: int = 3,
    ) -> None:
        super().__init__()
        self.input_dim = lookback * n_features
        self.horizon = horizon
        self.blocks = nn.ModuleList(
            [
                NBEATSBlock(
                    self.input_dim,
                    horizon,
                    hidden=hidden,
                    depth=depth,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x.flatten(start_dim=1)
        forecast = torch.zeros(
            x.size(0),
            self.horizon,
            device=x.device,
            dtype=x.dtype,
        )

        for block in self.blocks:
            backcast, contribution = block(residual)
            residual = residual - backcast
            forecast = forecast + contribution

        return forecast


class DiagonalSSMLayer(nn.Module):
    """A clarity-first diagonal linear state-space layer.

    The continuous dynamics use a negative diagonal A matrix, then apply an
    exact zero-order-hold discretization before scanning over the sequence.
    """

    def __init__(self, d_model: int, d_state: int = 16) -> None:
        super().__init__()
        rates = torch.logspace(-1, 1, d_state, dtype=torch.float32)
        self.A_log = nn.Parameter(rates.log().repeat(d_model, 1))
        self.B = nn.Parameter(torch.randn(d_model, d_state) / d_state**0.5)
        self.C = nn.Parameter(torch.randn(d_model, d_state) / d_state**0.5)
        self.D = nn.Parameter(torch.ones(d_model))
        self.log_dt = nn.Parameter(torch.full((d_model,), -3.0))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        batch, length, d_model = u.shape
        d_state = self.A_log.size(1)
        state = u.new_zeros(batch, d_model, d_state)

        # A is always negative, so every continuous mode is stable.
        A = -torch.exp(self.A_log)
        dt = torch.nn.functional.softplus(self.log_dt)
        A_bar = torch.exp(dt[:, None] * A)
        B_bar = ((A_bar - 1.0) / A) * self.B

        outputs = []
        for step in range(length):
            input_t = u[:, step, :]
            state = A_bar.unsqueeze(0) * state
            state = state + B_bar.unsqueeze(0) * input_t.unsqueeze(-1)
            output_t = (state * self.C.unsqueeze(0)).sum(dim=-1)
            output_t = output_t + self.D * input_t
            outputs.append(output_t)

        return torch.stack(outputs, dim=1)


class StateSpaceForecaster(nn.Module):
    """A fixed-dynamics state-space forecaster for [B, L, F] inputs."""

    def __init__(
        self,
        n_features: int,
        horizon: int,
        d_model: int = 64,
        d_state: int = 16,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(n_features, d_model)
        self.ssm = DiagonalSSMLayer(d_model, d_state=d_state)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = self.ssm(self.input_projection(x))
        final_state = self.norm(sequence[:, -1, :])
        return self.head(self.dropout(final_state))


class SelectiveSSMLayer(nn.Module):
    """An educational selective scan inspired by the original Mamba paper."""

    def __init__(self, d_model: int, d_state: int = 16) -> None:
        super().__init__()
        rates = torch.logspace(-1, 1, d_state, dtype=torch.float32)
        self.A_log = nn.Parameter(rates.log().repeat(d_model, 1))
        self.delta_projection = nn.Linear(d_model, d_model)
        nn.init.normal_(self.delta_projection.weight, std=0.02)
        nn.init.constant_(self.delta_projection.bias, -3.0)
        self.B_projection = nn.Linear(d_model, d_state)
        self.C_projection = nn.Linear(d_model, d_state)
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        batch, length, d_model = u.shape
        d_state = self.A_log.size(1)
        state = u.new_zeros(batch, d_model, d_state)
        A = -torch.exp(self.A_log)
        outputs = []

        for step in range(length):
            input_t = u[:, step, :]

            # Unlike the fixed SSM, these discretization and read/write terms
            # depend on the current token (time step).
            delta_t = torch.nn.functional.softplus(
                self.delta_projection(input_t)
            )
            B_t = self.B_projection(input_t)
            C_t = self.C_projection(input_t)

            A_bar = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))
            B_bar = ((A_bar - 1.0) / A.unsqueeze(0))
            B_bar = B_bar * B_t.unsqueeze(1)

            state = A_bar * state + B_bar * input_t.unsqueeze(-1)
            output_t = (state * C_t.unsqueeze(1)).sum(dim=-1)
            output_t = output_t + self.D * input_t
            outputs.append(output_t)

        return torch.stack(outputs, dim=1)


class MambaBlock(nn.Module):
    """A portable Mamba-style block with convolution, selection, and gating."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expansion: int = 2,
        conv_kernel: int = 4,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        d_inner = expansion * d_model
        self.norm = nn.LayerNorm(d_model)
        self.input_projection = nn.Linear(d_model, 2 * d_inner)
        self.depthwise_conv = nn.Conv1d(
            d_inner,
            d_inner,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,
            groups=d_inner,
        )
        self.ssm = SelectiveSSMLayer(d_inner, d_state=d_state)
        self.output_projection = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        content, gate = self.input_projection(self.norm(x)).chunk(2, dim=-1)

        length = content.size(1)
        content = self.depthwise_conv(content.transpose(1, 2))
        content = content[:, :, :length].transpose(1, 2)
        content = torch.nn.functional.silu(content)

        content = self.ssm(content)
        content = content * torch.nn.functional.silu(gate)
        content = self.output_projection(content)
        return residual + self.dropout(content)


class MambaStyleForecaster(nn.Module):
    """A dependency-free, educational Mamba-style direct forecaster."""

    def __init__(
        self,
        n_features: int,
        horizon: int,
        d_model: int = 64,
        d_state: int = 16,
        n_blocks: int = 2,
        expansion: int = 2,
        conv_kernel: int = 4,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(n_features, d_model)
        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    expansion=expansion,
                    conv_kernel=conv_kernel,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = self.input_projection(x)
        for block in self.blocks:
            sequence = block(sequence)
        final_state = self.norm(sequence[:, -1, :])
        return self.head(final_state)


def smoke_test() -> None:
    batch, lookback, n_features, horizon = 2, 168, 1, 24
    x = torch.randn(batch, lookback, n_features)
    models = {
        "linear": LinearForecaster(lookback, n_features, horizon),
        "mlp": MLPForecaster(lookback, n_features, horizon),
        "lstm": LSTMForecaster(n_features, horizon),
        "gru": GRUForecaster(n_features, horizon),
        "cnn1d": CNN1DForecaster(n_features, horizon),
        "tcn": TCNForecaster(n_features, horizon),
        "transformer": TransformerForecaster(lookback, n_features, horizon),
        "patch_transformer": PatchTransformerForecaster(
            lookback, n_features, horizon
        ),
        "nbeats_style": NBEATSForecaster(lookback, n_features, horizon),
        "state_space": StateSpaceForecaster(n_features, horizon),
        "mamba_style": MambaStyleForecaster(n_features, horizon),
    }

    for name, model in models.items():
        y = model(x)
        assert y.shape == (batch, horizon), (name, y.shape)
        print(f"{name:>18}: {tuple(y.shape)}")


if __name__ == "__main__":
    smoke_test()
