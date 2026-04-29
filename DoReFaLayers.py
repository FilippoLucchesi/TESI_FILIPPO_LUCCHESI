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
        self.trace_kwargs.pop('num_bits', None)

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
        self.trace_kwargs.pop('num_bits', None)

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

# fully connected net, MNIST, see if it converges (alpha polarizes?), 8/16 bit quantization

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