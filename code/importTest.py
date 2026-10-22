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