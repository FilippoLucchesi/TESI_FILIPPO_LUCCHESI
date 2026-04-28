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
	levels = (2 ** int(k)) - 1
	if levels <= 0:
		return torch.zeros_like(x)
	x_q = torch.round(x * levels) / levels
	return x + (x_q - x).detach()


def dorefa_weight(weight: torch.Tensor, k: int) -> torch.Tensor:
	w = torch.tanh(weight)
	max_val = w.detach().abs().max()
	w_norm = w / (2 * max_val + 1e-8) + 0.5
	return 2 * _quantize_ste(w_norm, k) - 1


def dorefa_activation(x: torch.Tensor, k: int) -> torch.Tensor:
	return _quantize_ste(torch.clamp(x, 0.0, 1.0), k)


class MutableDoReFaConv2d(MutableConv2d):
	"""MutableConv2d with DoReFa quantization in forward."""

	def __init__(self, *args, num_bits: int = 4, **kwargs):
		super().__init__(*args, **kwargs)
		self.num_bits = int(num_bits)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		w_q = dorefa_weight(self.weight, self.num_bits)
		x_q = dorefa_activation(x, self.num_bits)
		return self._conv_forward(x_q, w_q, self.bias)

	def freeze(self, sample):
		self.validate(sample)
		args, kwargs = self.freeze_init_arguments(sample, *self.trace_args, **self.trace_kwargs)
		frozen = MutableDoReFaConv2d(*args, **kwargs)
		frozen.load_state_dict(self.state_dict(), strict=False)
		return frozen


class MutableDoReFaLinear(MutableLinear):
	"""MutableLinear with DoReFa quantization in forward."""

	def __init__(self, *args, num_bits: int = 4, **kwargs):
		super().__init__(*args, **kwargs)
		self.num_bits = int(num_bits)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		w_q = dorefa_weight(self.weight, self.num_bits)
		x_q = dorefa_activation(x, self.num_bits)
		return F.linear(x_q, w_q, self.bias)

	def freeze(self, sample):
		self.validate(sample)
		args, kwargs = self.freeze_init_arguments(sample, *self.trace_args, **self.trace_kwargs)
		frozen = MutableDoReFaLinear(*args, **kwargs)
		frozen.load_state_dict(self.state_dict(), strict=False)
		return frozen


class CustomDARTSSpace(ModelSpace):
	"""Patched quantization-aware DARTS search space.

	This corresponds to a DOF-10 style space:
	- LayerChoice on first 2 blocks
	- channel choices for layer1_out and layer2_out
	- remaining conv stack fixed structure
	- all learnable conv/linear ops quantized via DoReFa wrappers
	"""

	def __init__(
		self,
		input_channels: int = 3,
		channels: int = 64,
		num_classes: int = 43,
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
		layer3_out = 16
		layer4_out = 16
		layer5_out = 16
		layer6_out = 16
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

		self.layers.append(
			nn.Sequential(
				nn.AvgPool2d(kernel_size=2, stride=1, padding=1),
				MutableDoReFaConv2d(layer2_out, layer3_out, kernel_size=3, padding=0, bias=False, num_bits=num_bits),
				MutableBatchNorm2d(layer3_out),
				MutableReLU(),
			)
		)

		self.layers.append(
			nn.Sequential(
				nn.AvgPool2d(kernel_size=2, stride=1, padding=1),
				MutableDoReFaConv2d(layer3_out, layer4_out, kernel_size=3, padding=0, bias=False, num_bits=num_bits),
				MutableBatchNorm2d(layer4_out),
				MutableReLU(),
			)
		)

		self.layers.append(
			nn.Sequential(
				nn.AvgPool2d(kernel_size=2, stride=1, padding=1),
				MutableDoReFaConv2d(layer4_out, layer5_out, kernel_size=3, padding=0, bias=False, num_bits=num_bits),
				MutableBatchNorm2d(layer5_out),
				MutableReLU(),
			)
		)

		self.layers.append(
			nn.Sequential(
				nn.AvgPool2d(kernel_size=2, stride=1, padding=1),
				MutableDoReFaConv2d(layer5_out, layer6_out, kernel_size=3, padding=0, bias=False, num_bits=num_bits),
				MutableBatchNorm2d(layer6_out),
				MutableReLU(),
			)
		)

		self.layers.append(
			nn.Sequential(
				nn.AvgPool2d(kernel_size=2, stride=1, padding=1),
				MutableDoReFaConv2d(layer6_out, layer7_out, kernel_size=3, padding=0, bias=False, num_bits=num_bits),
				MutableBatchNorm2d(layer7_out),
				MutableReLU(),
			)
		)

		self.pool = nn.AdaptiveAvgPool2d((3, 3))
		feature1 = 32
		feature2 = 32
		feature3 = 32
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

