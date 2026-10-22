import sys
import types

nni = types.ModuleType('nni')
nni.mutable = types.ModuleType('nni.mutable')
nni.mutable.Categorical = object
nni.nas = types.ModuleType('nni.nas')
nni.nas.nn = types.ModuleType('nni.nas.nn')
nni.nas.nn.pytorch = types.ModuleType('nni.nas.nn.pytorch')
nni.nas.nn.pytorch.ModelSpace = object
nni.nas.nn.pytorch.MutableConv2d = object
nni.nas.nn.pytorch.MutableLinear = object
sys.modules['nni'] = nni
sys.modules['nni.mutable'] = nni.mutable
sys.modules['nni.nas'] = nni.nas
sys.modules['nni.nas.nn'] = nni.nas.nn
sys.modules['nni.nas.nn.pytorch'] = nni.nas.nn.pytorch

import torch
from DoReFaLayers import _quantize_ste

x = torch.tensor([0.1, 0.4, 0.9], requires_grad=True)
weights = torch.tensor([0.2, 0.6, -0.4])
loss = (_quantize_ste(x, 2) * weights).sum()
loss.backward()
print('grad', x.grad.tolist())