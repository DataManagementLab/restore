import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np


class MaskedLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

        self.masked_weight = None

    def set_mask(self, mask):
        """Accepts a mask of shape [in_features, out_features]."""
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input):
        if self.masked_weight is None:
            return F.linear(input, self.mask * self.weight, self.bias)
        else:
            # ~17% speedup for Prog Sampling.
            return F.linear(input, self.masked_weight, self.bias)


class MaskedResidualBlock(nn.Module):

    def __init__(self, in_features, out_features, activation):
        assert in_features == out_features, [in_features, out_features]
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(MaskedLinear(in_features, out_features, bias=True))
        self.layers.append(MaskedLinear(in_features, out_features, bias=True))
        self.activation = activation

    def set_mask(self, mask):
        self.layers[0].set_mask(mask)
        self.layers[1].set_mask(mask)

    def forward(self, input):
        out = input
        out = self.activation(out)
        out = self.layers[0](out)
        out = self.activation(out)
        out = self.layers[1](out)
        return input + out


class PartialMaskedLinear(MaskedLinear):
    """
    Similar to masked embedding but allows to add unmasked embedding
    """

    def __init__(self, in_features, unmasked_in_features, out_features, bias=True):
        MaskedLinear.__init__(self, in_features, out_features, bias)

        self.unmasked_weight = Parameter(torch.Tensor(out_features, unmasked_in_features))
        self.unmasked_bias = Parameter(torch.Tensor(out_features))

    def forward(self, input, unmasked_input):
        output_masked = F.linear(input, self.mask * self.weight, self.bias)
        output_unmasked = F.linear(unmasked_input, self.unmasked_weight, self.unmasked_bias)
        return output_masked + output_unmasked


class PartialMaskedResidualBlock(MaskedResidualBlock):

    def __init__(self, in_features, unmasked_in_features, out_features, activation):
        MaskedResidualBlock.__init__(self, in_features, out_features, activation)
        self.layers = nn.ModuleList()
        self.layers.append(MaskedLinear(in_features, out_features, bias=True))
        self.layers.append(PartialMaskedLinear(in_features, unmasked_in_features, out_features, bias=True))

    def forward(self, input, unmasked_input):
        out = input
        out = self.activation(out)
        out = self.layers[0](out)
        out = self.activation(out)
        out = self.layers[1](out, unmasked_input)
        return input + out
