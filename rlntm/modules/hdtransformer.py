import torch
import math
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, ModuleList, LayerNorm
from torch.nn.modules.transformer import TransformerEncoderLayer
from typing import Optional, Any, List

class HDTransformerEncoderLayer(Module):

    def __init__(self, n_dims, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(HDTransformerEncoderLayer, self).__init__()

        self.layers = torch.nn.ModuleList([TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation) for _ in range(n_dims)])

    def forward(self, src: Tensor, src_mask: Optional[List[Tensor]] = None, src_key_padding_mask: Tensor = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: List of masks for the src sequence. None values are ignored in the list. (optional).
            src_key_padding_mask: List of masks for the src keys per batch. None values are ignored in the list. (optional).

        Shape:
            src: (N, d_1, ..., d_n, E)
            src_masks: (d_1, d_1), ..., (d_n, d_n)
            src_key_padding_mask: (N, d_1, ..., d_n)
        """
        if src_mask is None:
            src_mask = [None for _ in range(len(list(src.size()[1:-1])))]

        for i in range(len(self.layers)):
            reshaped_src = src.transpose(-2, - 2 - i)
            msk = src_mask[i]
            keypmask = None
            if src_key_padding_mask is not None:
                keypmask = src_key_padding_mask.transpose(- 1, - 1 - i).reshape(-1, src_key_padding_mask.size(- 1 - i))

            x = self.layers[i](reshaped_src.reshape(-1, src.size(-2 - i), src.size(-1)).transpose(0, 1), src_mask=msk, src_key_padding_mask=keypmask).transpose(0,1)
            x = x.view(*reshaped_src.size())
            src = x.transpose(-2, - 2 - i)

        return src

class HDTransformerEncoder(Module):
    __constants__ = ['norm']

    def __init__(self, n_dims, d_model, nhead, num_layers=6, dim_feedforward=2048, dropout=0.1, activation="relu", layer_norm_eps = 1e-5):
        super(HDTransformerEncoder, self).__init__()
        self.layers = []
        for i in range(num_layers):
            self.layers.append(HDTransformerEncoderLayer(n_dims, d_model, nhead, dim_feedforward, dropout, activation))

        self.layers = nn.ModuleList(self.layers)

        self.num_layers = num_layers
        self.norm = LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.
        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

def test():
    trans = HDTransformerEncoderLayer(4, 512, 8)
    in_val = torch.randn(6, 2, 3, 4, 5, 512)
    data = trans(in_val)

    trans2 = HDTransformerEncoderLayer(4, 512, 8)
    data2 = trans2(in_val)
    import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    test()
