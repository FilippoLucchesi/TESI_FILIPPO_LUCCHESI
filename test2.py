import torch
import torch.nn as nn
import torch.nn.functional as F
from nni.mutable import Categorical, ensure_frozen
from nni.nas.nn.pytorch import (
    LayerChoice,
    ModelSpace,
    MutableBatchNorm2d,
    MutableConv2d,
    MutableLinear,
    MutableModule,
    MutableReLU,
)
from nni.nas.space import model_context

def _quantize_ste(x: torch.Tensor, k: int) -> torch.Tensor:
    levels = (2 ** int(k)) - 1
    if levels <= 0:
        return torch.zeros_like(x)
    x_q = torch.round(x * levels) / levels
    return x + (x_q - x).detach()


def quantize_weights_dorefa(weight: torch.Tensor, k_bits: int) -> torch.Tensor:
    weight_tanh = torch.tanh(weight)
    max_val = weight_tanh.detach().abs().max()
    weight_norm = weight_tanh / (2 * max_val + 1e-8) + 0.5
    weight_q = _quantize_ste(weight_norm, k_bits)
    return 2 * weight_q - 1


def quantize_inputs_dorefa(x: torch.Tensor, k_bits: int) -> torch.Tensor:
    return _quantize_ste(torch.clamp(x, 0.0, 1.0), k_bits)


class MutableDoReFaConv2d(MutableModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        bit_choices: tuple[int, ...] = (2, 4, 8),
        bit_label: str = 'dorefa_bits',
    ) -> None:
        super().__init__()
        self._init_args = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            bit_choices=bit_choices,
            bit_label=bit_label,
        )

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bitwidth = Categorical(bit_choices, label=bit_label)
        self.add_mutable(self.bitwidth)
        self._dry_run_bits = ensure_frozen(self.bitwidth, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bits = ensure_frozen(self.bitwidth, strict=False)
        x_q = quantize_inputs_dorefa(x, bits)
        w_q = quantize_weights_dorefa(self.conv.weight, bits)
        return F.conv2d(
            x_q,
            w_q,
            self.conv.bias,
            self.conv.stride,
            self.conv.padding,
            self.conv.dilation,
            self.conv.groups,
        )

    def freeze(self, sample: dict[str, int]) -> nn.Module:
        self.validate(sample)
        with model_context(sample):
            frozen_layer = MutableDoReFaConv2d(**self._init_args)
        frozen_layer.load_state_dict(self.state_dict(), strict=False)
        return frozen_layer


class CustomDARTSSpace(ModelSpace):
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 43,
        verbose: int = 0,
        drop_path_prob: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.drop_path_prob = drop_path_prob
        self.verbose = verbose

        channel_plan = [16, 32, 32, 64, 64, 64, 22]

        self.preliminary_layer = nn.Conv2d(input_channels, channel_plan[0], kernel_size=3, padding=0, bias=False)
        self.layer0_bn = nn.BatchNorm2d(channel_plan[0])
        self.layer0_relu = nn.ReLU(inplace=True)

        in_channels = channel_plan[0]
        for idx, out_channels in enumerate(channel_plan[1:], start=1):
            use_dorefa = idx in {1, 3, 5}

            def make_conv(in_ch: int, out_ch: int, layer_idx: int = idx) -> nn.Module:
                if use_dorefa:
                    return MutableDoReFaConv2d(
                        in_ch,
                        out_ch,
                        kernel_size=3,
                        padding=1,
                        bias=True,
                        bit_choices=(2, 4, 8),
                        bit_label=f'dorefa_bits_l{layer_idx}',
                    )
                return MutableConv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)

            layer = LayerChoice(
                [
                    nn.Sequential(
                        nn.AvgPool2d(kernel_size=2, stride=1, padding=1),
                        make_conv(in_channels, out_channels),
                        MutableBatchNorm2d(out_channels),
                        MutableReLU(),
                    ),
                    nn.Sequential(
                        make_conv(in_channels, out_channels),
                        nn.AvgPool2d(kernel_size=2, stride=1, padding=1),
                        MutableBatchNorm2d(out_channels),
                        MutableReLU(),
                    ),
                ],
                label=f'layer_{idx}',
            )
            self.layers.append(layer)
            in_channels = out_channels

        self.pool = nn.AdaptiveAvgPool2d((3, 3))
        self.fc1 = MutableLinear(22 * 3 * 3, 64)
        self.fc2 = MutableLinear(64, 32)
        self.fc3 = MutableLinear(32, 32)
        self.relu = nn.ReLU()
        self.classifier = MutableLinear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preliminary_layer(x)
        x = self.layer0_bn(x)
        x = self.layer0_relu(x)

        if self.verbose == 1:
            print(f'After preliminary layer: {x.shape}')

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.verbose == 1:
                print(f'After layer {i + 1}: {x.shape}')
            if i in {1, 3, 5}:
                x = nn.AvgPool2d(kernel_size=2, stride=2)(x)
                if self.verbose == 1:
                    print(f'After avg pooling: {x.shape}')

        x = self.pool(x)
        if self.verbose == 1:
            print(f'After adaptive pooling: {x.shape}')

        x = torch.flatten(x, 1)
        if self.verbose == 1:
            print(f'After flattening: {x.shape}')

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.classifier(x)

        if self.verbose == 1:
            print(f'After classifier: {x.shape}')

        return x


if __name__ == '__main__':
    model_space = CustomDARTSSpace()
    print('Search-space keys:', sorted(model_space.simplify().keys()))
    out = model_space(torch.rand(2, 3, 48, 48))
    print('Output shape:', tuple(out.shape))