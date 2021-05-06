import torch
import math
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, ModuleList, LayerNorm, Linear, Dropout, MultiheadAttention
from torch.nn.modules.transformer import TransformerEncoderLayer
from typing import Optional, Any, List
import torch.nn.functional as F

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class HDTransformerEncoderLayer(Module):

    def __init__(self, n_dims, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", max_window=4000000000, stride=20):
        super(HDTransformerEncoderLayer, self).__init__()

        self.self_attns = torch.nn.ModuleList([MultiheadAttention(d_model, nhead, dropout=dropout) for _ in range(n_dims)])
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.max_window = max_window
        self.stride = stride

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

        src2 = src
        for i in range(len(self.self_attns)):
            reshaped_src = src.transpose(-2, - 2 - i)
            msk = src_mask[i]
            keypmask = None
            if src_key_padding_mask is not None:
                keypmask = src_key_padding_mask.transpose(- 1, - 1 - i).reshape(-1, src_key_padding_mask.size(- 1 - i))


            enc_in = reshaped_src.reshape(-1, src.size(- 2 - i), src.size(-1)).transpose(0, 1)
            if enc_in.size(0) > self.max_window:
                enc_in_unfolded = enc_in.unfold(0, self.max_window, self.stride).unsqueeze(0).transpose(0, -1).squeeze()
                unfold_size = enc_in_unfolded.size()
                if keypmask is not None:
                    keypmask = keypmask.unfold(-1, self.max_window, self.stride).transpose(0, 1).reshape(-1, self.max_window)

                x = self.self_attns[i](enc_in_unfolded.reshape(self.max_window, -1, enc_in_unfolded.size(-1)), attn_mask=msk, key_padding_mask=keypmask)
                x = x.reshape(unfold_size)

                # Sum overlapping windows
                x_big = torch.zeros([enc_in.size(0)] + [self.max_window // self.stride] + list(x.size())[2:], device=enc_in.device)
                x_big_stacked = x_big.reshape([-1] + list(x_big.size())[2:])
                indices = torch.tensor(sum([[(self.max_window // self.stride) * (i + self.stride * offset) + (offset % (self.max_window // self.stride)) for i in range(self.max_window)] for offset in range(x.size(1))], []), device=x_big_stacked.device)
                x_big_stacked = x_big_stacked.index_add(0, indices, x.reshape([-1] + list(x.size())[2:]))
                x_big = x_big_stacked.reshape(x_big.size())
                x = x_big.sum(1).transpose(0, 1)

                x = x.view(*reshaped_src.size())
                src2 = x.transpose(-2, - 2 - i)
            else:
                # Eliminate nans
                if keypmask is not None:
                    kpmsk = keypmask.clone()
                    kpmsk[keypmask.sum(-1) == keypmask.size(-1)] = False
                    x = self.self_attns[i](enc_in, enc_in, enc_in, attn_mask=msk, key_padding_mask=kpmsk)[0].transpose(0,1)
                    x = x * (~(keypmask.sum(-1) == keypmask.size(-1))).float().unsqueeze(-1).unsqueeze(-1)
                else:
                    x = self.self_attns[i](enc_in, enc_in, enc_in, attn_mask=msk, key_padding_mask=keypmask)[0].transpose(0,1)

                x = x.view(*reshaped_src.size())
                src2 = src2 + self.dropout1(x.transpose(-2, - 2 - i))

        src = self.norm1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

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

def positionalencoding3d(d_model, height, width, depth):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    """
    pe = torch.zeros(d_model, height, width, depth)
    # Each dimension use half of d_model
    d_model = 2 * int(d_model / 6)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pos_z = torch.arange(0., depth).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).unsqueeze(-1).repeat(1, height, 1, depth)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).unsqueeze(-1).repeat(1, height, 1, depth)
    pe[d_model:2*d_model:2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).unsqueeze(-1).repeat(1, 1, width, depth)
    pe[d_model + 1:2*d_model:2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).unsqueeze(-1).repeat(1, 1, width, depth)
    pe[2*d_model:3*d_model:2, :, :] = torch.sin(pos_z * div_term).transpose(0, 1).unsqueeze(-2).unsqueeze(-2).repeat(1, height, width, 1)
    pe[2*d_model + 1:3*d_model:2, :, :] = torch.cos(pos_z * div_term).transpose(0, 1).unsqueeze(-2).unsqueeze(-2).repeat(1, height, width, 1)

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
