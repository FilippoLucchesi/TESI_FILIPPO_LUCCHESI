import torch
import torch.nn as nn
import torch.nn.functional as F

import nni
from nni.nas.nn.pytorch import (
	LayerChoice,
	ModelSpace,
	MutableBatchNorm2d,
	MutableConv2d,
	MutableLinear,
	MutableReLU,
)


def _quantize_ste(x: torch.Tensor, k: int) -> torch.Tensor:
    """Quantize tensor x to k bits using a straight-through estimator (STE).

    Parameters
    ----------
    x : torch.Tensor
        Input tensor to quantize.
    k : int
        Number of quantization bits.
    """
    # Number of representable integer steps in [0, 1].
    levels = (2 ** int(k)) - 1
    if levels <= 0:
        # Degenerate case (k <= 0): return all zeros.
        return torch.zeros_like(x)

    # Forward quantization (discrete values).
    x_q = torch.round(x * levels) / levels
    # STE/quantization trick: forward uses x_q, backward gradient flows as if identity on x.
    return x + (x_q - x).detach()


def dorefa_weight(weight: torch.Tensor, k: int) -> torch.Tensor:
    """DoReFa quantization for convolution weights.

    Steps:
    1) tanh squash,
    2) normalize to [0, 1],
    3) quantize to k bits,
    4) map back to [-1, 1].
    """
    # Bound raw weights to a stable range.
    weight_tanh = torch.tanh(weight)

    # Scale factor (detached to avoid gradient flowing through max operation).
    max_val = weight_tanh.detach().abs().max()

    # Map from (approximately) [-1, 1] to [0, 1].
    weight_norm = weight_tanh / (2 * max_val + 1e-8) + 0.5

    # Quantize normalized weights.
    weight_q = 2 * _quantize_ste(weight_norm, k) - 1
    return weight_q


def dorefa_activation(x: torch.Tensor, k: int) -> torch.Tensor:
    """DoReFa-style activation quantization.

    Activations are clipped to [0, 1] first, then quantized to k bits.
    """
    return _quantize_ste(torch.clamp(x, 0.0, 1.0), k)

# MUTABLE: base class for every class representing a search space.
class MutableDoReFaConv2d(MutableConv2d):
    """NNI MutableConv2d + DoReFa quantization.

    Structural parameters (kernel_size, stride, etc.) can be mutable exactly
    like MutableConv2d, while quantization bitwidth is a fixed parameter.
    """

    def __init__( 
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
        num_bits: int = 4,
    ) -> None:
        # Initialize the parent mutable convolution. Any mutable structural args
        # passed here are tracked by MutableConv2d.
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        # Fixed quantization precision configured outside NAS.
        self.num_bits = int(num_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize both weights and inputs (activations from previous layer).
        w_q = dorefa_weight(self.weight, self.num_bits)
        x_q = dorefa_activation(x, self.num_bits)

        # Reuse Conv2d internal forward (handles padding mode/groups/etc.).
        return self._conv_forward(x_q, w_q, self.bias)

    # FREEZE: convert from mutable search space to deterministic model.
    # WORKFLOW: FREEZE a model, then train it, then checkpoint or export it.

    def freeze(self, sample: dict[str, int]) -> nn.Module:

        # Confirm that provided architecture sample contains legal choices.
        self.validate(sample)

        # Freeze structural mutables from MutableConv2d trace args.
        args, kwargs = self.freeze_init_arguments(sample, *self.trace_args, **self.trace_kwargs)

        # Build a deterministic clone.
        frozen_layer = MutableDoReFaConv2d(*args, **kwargs)

        # Copy learned float weights/bias.
        frozen_layer.load_state_dict(self.state_dict(), strict=False)

        return frozen_layer


class MutableDoReFaLinear(MutableLinear):
    """NNI MutableLinear + DoReFa quantization.

    Structural parameters (in_features/out_features) can be mutable exactly
    like MutableLinear, while quantization bitwidth is a fixed parameter.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        num_bits: int = 4,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.num_bits = int(num_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = dorefa_weight(self.weight, self.num_bits)
        x_q = dorefa_activation(x, self.num_bits)
        return F.linear(x_q, w_q, self.bias)

    def freeze(self, sample: dict[str, int]) -> nn.Module:
        self.validate(sample)
        args, kwargs = self.freeze_init_arguments(sample, *self.trace_args, **self.trace_kwargs)
        frozen_layer = MutableDoReFaLinear(*args, **kwargs)
        frozen_layer.load_state_dict(self.state_dict(), strict=False)
        return frozen_layer


class CustomDARTSSpace(ModelSpace):
	"""Quantization-aware DARTS search space with DoReFa quantization.

	This corresponds to a flexible search space with:
	- LayerChoice on first 2 blocks (pool/conv ordering)
	- channel choices for layers 1-6
	- all learnable conv/linear ops quantized via DoReFa wrappers
	"""

	def __init__(
		self,
		input_channels: int = 3,
		channels: int = 64,
		num_classes: int = 10,
		layers: int = 7,
		verbose: int = 0,
		drop_path_prob: float = 0.1,
		num_bits: int = 4,
	):
		super().__init__()
		self.layers = nn.ModuleList()
		self.drop_path_prob = drop_path_prob
		self.verbose = verbose

		layer0_out = 16
		layer1_out = nni.choice("layer1_out_channels", [16, 32, 64])
		layer2_out = nni.choice("layer2_out_channels", [16, 32, 64])
		layer3_out = nni.choice("layer3_out_channels", [16, 32, 64])
		layer4_out = nni.choice("layer4_out_channels", [16, 32, 64])
		layer5_out = nni.choice("layer5_out_channels", [16, 32, 64])
		layer6_out = nni.choice("layer6_out_channels", [16, 32, 64])
		layer7_out = 22

		# Quantize first stem conv as well for full QAT consistency.
		self.preliminary_layer = MutableDoReFaConv2d(
			input_channels,
			layer0_out,
			kernel_size=3,
			padding=0,
			bias=False,
			num_bits=num_bits,
		)
		self.layer0_bn = MutableBatchNorm2d(layer0_out)
		self.layer0_relu = MutableReLU()

		layer1 = LayerChoice(
			[
				nn.Sequential(
					nn.AvgPool2d(kernel_size=2, stride=1, padding=1),
					MutableDoReFaConv2d(
						layer0_out,
						layer1_out,
						kernel_size=3,
						bias=False,
						num_bits=num_bits,
					),
					MutableBatchNorm2d(layer1_out),
					MutableReLU(),
				),
				nn.Sequential(
					MutableDoReFaConv2d(
						layer0_out,
						layer1_out,
						kernel_size=3,
						bias=False,
						num_bits=num_bits,
					),
					nn.AvgPool2d(kernel_size=2, stride=1, padding=1),
					MutableBatchNorm2d(layer1_out),
					MutableReLU(),
				),
			],
			label="layer_1",
		)
		self.layers.append(layer1)

		layer2 = LayerChoice(
			[
				nn.Sequential(
					nn.AvgPool2d(kernel_size=2, stride=1, padding=1),
					MutableDoReFaConv2d(
						layer1_out,
						layer2_out,
						kernel_size=3,
						bias=False,
						num_bits=num_bits,
					),
					MutableBatchNorm2d(layer2_out),
					MutableReLU(),
				),
				nn.Sequential(
					MutableDoReFaConv2d(
						layer1_out,
						layer2_out,
						kernel_size=3,
						bias=False,
						num_bits=num_bits,
					),
					nn.AvgPool2d(kernel_size=2, stride=1, padding=1),
					MutableBatchNorm2d(layer2_out),
					MutableReLU(),
				),
			],
			label="layer_2",
		)
		self.layers.append(layer2)

		layer3 = LayerChoice(
			[
				nn.Sequential(
					nn.AvgPool2d(kernel_size=2, stride=1, padding=1),
					MutableDoReFaConv2d(
						layer2_out,
						layer3_out,
						kernel_size=3,
						bias=False,
						num_bits=num_bits,
					),
					MutableBatchNorm2d(layer3_out),
					MutableReLU(),
				),
				nn.Sequential(
					MutableDoReFaConv2d(
						layer2_out,
						layer3_out,
						kernel_size=3,
						bias=False,
						num_bits=num_bits,
					),
					nn.AvgPool2d(kernel_size=2, stride=1, padding=1),
					MutableBatchNorm2d(layer3_out),
					MutableReLU(),
				),
			],
			label="layer_3",
		)
		self.layers.append(layer3)

		layer4 = LayerChoice(
			[
				nn.Sequential(
					nn.AvgPool2d(kernel_size=2, stride=1, padding=1),
					MutableDoReFaConv2d(
						layer3_out,
						layer4_out,
						kernel_size=3,
						bias=False,
						num_bits=num_bits,
					),
					MutableBatchNorm2d(layer4_out),
					MutableReLU(),
				),
				nn.Sequential(
					MutableDoReFaConv2d(
						layer3_out,
						layer4_out,
						kernel_size=3,
						bias=False,
						num_bits=num_bits,
					),
					nn.AvgPool2d(kernel_size=2, stride=1, padding=1),
					MutableBatchNorm2d(layer4_out),
					MutableReLU(),
				),
			],
			label="layer_4",
		)
		self.layers.append(layer4)

		layer5 = LayerChoice(
			[
				nn.Sequential(
					nn.AvgPool2d(kernel_size=2, stride=1, padding=1),
					MutableDoReFaConv2d(
						layer4_out,
						layer5_out,
						kernel_size=3,
						bias=False,
						num_bits=num_bits,
					),
					MutableBatchNorm2d(layer5_out),
					MutableReLU(),
				),
				nn.Sequential(
					MutableDoReFaConv2d(
						layer4_out,
						layer5_out,
						kernel_size=3,
						bias=False,
						num_bits=num_bits,
					),
					nn.AvgPool2d(kernel_size=2, stride=1, padding=1),
					MutableBatchNorm2d(layer5_out),
					MutableReLU(),
				),
			],
			label="layer_5",
		)
		self.layers.append(layer5)

		layer6 = LayerChoice(
			[
				nn.Sequential(
					nn.AvgPool2d(kernel_size=2, stride=1, padding=1),
					MutableDoReFaConv2d(
						layer5_out,
						layer6_out,
						kernel_size=3,
						bias=False,
						num_bits=num_bits,
					),
					MutableBatchNorm2d(layer6_out),
					MutableReLU(),
				),
				nn.Sequential(
					MutableDoReFaConv2d(
						layer5_out,
						layer6_out,
						kernel_size=3,
						bias=False,
						num_bits=num_bits,
					),
					nn.AvgPool2d(kernel_size=2, stride=1, padding=1),
					MutableBatchNorm2d(layer6_out),
					MutableReLU(),
				),
			],
			label="layer_6",
		)
		self.layers.append(layer6)

		layer7 = LayerChoice(
			[
				nn.Sequential(
					nn.AvgPool2d(kernel_size=2, stride=1, padding=1),
					MutableDoReFaConv2d(
						layer6_out,
						layer7_out,
						kernel_size=3,
						bias=False,
						num_bits=num_bits,
					),
					MutableBatchNorm2d(layer7_out),
					MutableReLU(),
				),
				nn.Sequential(
					MutableDoReFaConv2d(
						layer6_out,
						layer7_out,
						kernel_size=3,
						bias=False,
						num_bits=num_bits,
					),
					nn.AvgPool2d(kernel_size=2, stride=1, padding=1),
					MutableBatchNorm2d(layer7_out),
					MutableReLU(),
				),
			],
			label="layer_7",
		)
		self.layers.append(layer7)

		self.pool = nn.AdaptiveAvgPool2d((3, 3))
		feature1 = nni.choice("feature1", [32, 64, 128])
		feature2 = nni.choice("feature2", [32, 64, 128])
		feature3 = nni.choice("feature3", [32, 64])
		self.fc1 = MutableDoReFaLinear(198, feature1, num_bits=num_bits)
		self.fc2 = MutableDoReFaLinear(feature1, feature2, num_bits=num_bits)
		self.fc3 = MutableDoReFaLinear(feature2, feature3, num_bits=num_bits)
		self.relu = nn.ReLU()
		self.classifier = MutableDoReFaLinear(feature3, num_classes, num_bits=num_bits)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.preliminary_layer(x)
		x = self.layer0_bn(x)
		x = self.layer0_relu(x)
		if self.verbose == 1:
			print(f"After preliminary layer: {x.shape}")

		for i, layer in enumerate(self.layers):
			x = layer(x)
			if self.verbose == 1:
				print(f"After layer {i + 1}: {x.shape}")
			if i in (1, 3, 6):
				x = nn.AvgPool2d(kernel_size=2, stride=2)(x)
				if self.verbose == 1:
					print(f"After avg pooling: {x.shape}")

		x = self.pool(x)
		if self.verbose == 1:
			print(f"After adaptive pooling: {x.shape}")

		x = torch.flatten(x, 1)
		if self.verbose == 1:
			print(f"After flattening: {x.shape}")

		x = self.fc1(x)
		x = self.relu(x)
		if self.verbose == 1:
			print(f"After fc1: {x.shape}")

		x = self.fc2(x)
		x = self.relu(x)
		if self.verbose == 1:
			print(f"After fc2: {x.shape}")

		x = self.fc3(x)
		x = self.relu(x)
		if self.verbose == 1:
			print(f"After fc3: {x.shape}")

		x = self.classifier(x)
		if self.verbose == 1:
			print(f"After classifier: {x.shape}")

		return x

	def set_drop_path_prob(self, drop_path_prob: float):
		self.drop_path_prob = drop_path_prob
		for layer in self.layers:
			if hasattr(layer, "set_drop_path_prob"):
				layer.set_drop_path_prob(drop_path_prob)

