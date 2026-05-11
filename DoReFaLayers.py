import torch
import torch.nn as nn
import torch.nn.functional as F

from nni.mutable import Categorical
from nni.nas.nn.pytorch import ModelSpace, MutableConv2d, MutableLinear

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


class FrozenDoReFaConv2d(nn.Conv2d):
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
        num_bits: int = 6,
    ) -> None:
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
        self.num_bits = int(num_bits)

    def forward(self, x: torch.Tensor, num_bits = None) -> torch.Tensor:
        bits = self.num_bits if num_bits is None else num_bits
        w_q = dorefa_weight(self.weight, bits)
        #x_q = dorefa_activation(x, self.num_bits)
        return self._conv_forward(x, w_q, self.bias)


class FrozenDoReFaLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        num_bits: int = 6,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.num_bits = int(num_bits)

    def forward(self, x: torch.Tensor, num_bits: int = None) -> torch.Tensor:
        bits = self.num_bits if num_bits is None else num_bits
        w_q = dorefa_weight(self.weight, bits)
        #x_q = dorefa_activation(x, self.num_bits)
        return F.linear(x, w_q, self.bias)

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
        kernel_size,
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
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        self.num_bits = int(num_bits)
        self.trace_kwargs.pop('num_bits', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = dorefa_weight(self.weight, self.num_bits)
        print("check Conv2d")
        # x_q = dorefa_activation(x, self.num_bits)
        return self._conv_forward(x, w_q, self.bias)

    def freeze(self, sample: dict[str, int]) -> nn.Module:
        self.validate(sample)
        args, kwargs = self.freeze_init_arguments(sample, *self.trace_args, **self.trace_kwargs)
        frozen_layer = FrozenDoReFaConv2d(*args, **kwargs)
        frozen_layer.load_state_dict(self.state_dict(), strict=False)
        return frozen_layer

class DoReFaLinear(nn.Module):
    def __init__(self, base_linear, num_bits):
        super().__init__()
        self.base = base_linear
        self.num_bits = num_bits

    def forward(self, x):
        return self.base(x)
    
class DoReFaConv2d(nn.Module):
    def __init__(self, base_conv, num_bits):
        super().__init__()
        self.base = base_conv
        self.num_bits = num_bits

    def forward(self, x):
        return self.base(x)
    '''ALTERNATIVE:
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Full precision path
        y_fp = self.base._conv_forward(x, self.base.weight, self.base.bias)

        # Quantized weights
        w_q = dorefa_weight(self.base.weight, self.num_bits)
        y_q = self.base._conv_forward(x, w_q, self.base.bias)

        # Blend (dual-path ready)
        alpha = self.quant_lambda
        return (1 - alpha) * y_fp + alpha * y_q
    '''


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
        self.trace_kwargs.pop('num_bits', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = dorefa_weight(self.weight, self.num_bits)
        # x_q = dorefa_activation(x, self.num_bits)
        print("check Linear")
        return F.linear(x, w_q, self.bias)

    def freeze(self, sample: dict[str, int]) -> nn.Module:
        self.validate(sample)
        args, kwargs = self.freeze_init_arguments(sample, *self.trace_args, **self.trace_kwargs)
        frozen_layer = FrozenDoReFaLinear(*args, **kwargs)
        frozen_layer.load_state_dict(self.state_dict(), strict=False)
        return frozen_layer

# fully connected net, MNIST, see if it converges (alpha polarizes?), 8/16 bit quantization


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_bits=6):
        super().__init__()

        self.depthwise = FrozenDoReFaConv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,   # key part
            bias=False,
            num_bits=num_bits
        )

        self.pointwise = FrozenDoReFaConv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            num_bits=num_bits
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, w_quant_bits=None):
        if w_quant_bits is not None:
            x = self.depthwise(x, w_quant_bits)
            x = self.pointwise(x, w_quant_bits)
        else:
            x = self.depthwise(x)
            x = self.pointwise(x)

        x = self.bn(x)
        x = self.relu(x)
        return x
    
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_ratio=4, num_bits=6):
        super().__init__()

        hidden = max(out_channels // bottleneck_ratio, 4)

        self.conv1 = FrozenDoReFaConv2d(
            in_channels,
            hidden,
            kernel_size=1,
            bias=False,
            num_bits=num_bits
        )

        self.conv2 = FrozenDoReFaConv2d(
            hidden,
            hidden,
            kernel_size=3,
            padding=1,
            bias=False,
            num_bits=num_bits
        )

        self.conv3 = FrozenDoReFaConv2d(
            hidden,
            out_channels,
            kernel_size=1,
            bias=False,
            num_bits=num_bits
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, w_quant_bits=None):

        if w_quant_bits is not None:
            x = self.conv1(x, w_quant_bits)
            x = self.relu(x)

            x = self.conv2(x, w_quant_bits)
            x = self.relu(x)

            x = self.conv3(x, w_quant_bits)
        else:
            x = self.conv1(x)
            x = self.relu(x)

            x = self.conv2(x)
            x = self.relu(x)

            x = self.conv3(x)

        x = self.bn(x)
        x = self.relu(x)

        return x

'''

if __name__ == '__main__':
    # Build the model search space.
    model_space = DoReFaToyNet()
    # Print all mutable dimensions known to NNI.
    print('Search space:', model_space.simplify())

    # Example fixed architecture sample used for freezing.
    sample = {
        'conv_kernel_size': 3,
        'conv_stride': 1,
    }

    # Convert search space model into a deterministic model.
    frozen_model = model_space.freeze(sample)
    print('Frozen with sample:', sample)

    # Quick forward-pass sanity test.
    inp = torch.rand(2, 1, 28, 28)
    out = frozen_model(inp)
    print('Output shape:', tuple(out.shape))

'''

'''
def _quantize_gradient(grad: torch.Tensor, k: int) -> torch.Tensor:
    """DoReFa gradient quantization with explicit noise term."""
    if k <= 0:
        return torch.zeros_like(grad)

    levels = (2 ** k) - 1

    # Compute per-sample max over all dims except batch
    # (assuming NCHW or similar)
    dims = list(range(1, grad.dim()))
    max_val = grad.abs().amax(dim=dims, keepdim=True)

    # Avoid division by zero
    max_val = torch.where(max_val == 0, torch.ones_like(max_val), max_val)

    # Normalize to [0,1]
    g = grad / (2 * max_val) + 0.5

    # Add noise: Uniform(-0.5, 0.5) scaled by quantization step
    noise = (torch.rand_like(g) - 0.5) / levels
    g = g + noise

    # Quantize to k bits in [0,1]
    g_q = torch.round(g * levels) / levels

    # Map back to [-1,1]
    g_q = g_q - 0.5
    g_q = 2 * max_val * g_q

    return g_q


class _DoReFaQuantizeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, k: int) -> torch.Tensor:
        ctx.k = int(k)

        levels = (2 ** ctx.k) - 1
        if levels <= 0:
            return torch.zeros_like(x)

        return torch.round(x * levels) / levels

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return _quantize_gradient(grad_output, ctx.k), None


def _quantize_ste(x: torch.Tensor, k: int) -> torch.Tensor:
    """Quantize tensor x to k bits and quantize gradients in backward pass.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor to quantize.
    k : int
        Number of quantization bits.
    """
    return _DoReFaQuantizeFn.apply(x, int(k))

    class DoReFaToyNet(ModelSpace):
    """Minimal model space using the custom quantized mutable convolution."""

    def __init__(self, num_bits: int = 4) -> None:
        super().__init__()

        # Structural mutables for the custom conv.
        # CATEGORICAL: explicit mutable object API, integrates better with freeze() and simplify()
        conv_kernel_size = Categorical([3, 5], label='conv_kernel_size')
        conv_stride = Categorical([1, 2], label='conv_stride')

        # Custom quantized mutable conv: structural mutables + fixed bitwidth.
        self.quant_conv = MutableDoReFaConv2d(
            1,
            8,
            kernel_size=conv_kernel_size,
            stride=conv_stride,
            padding=2,
            num_bits=num_bits,
        )

        # Simple classification head.
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            MutableDoReFaLinear(8, 10, num_bits=num_bits),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant_conv(x)
        return self.head(x)
'''