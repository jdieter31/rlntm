import torch.nn as nn
import math
import torch
from torch import Tensor
from torch.nn import Module, ModuleList, LayerNorm, Linear, Dropout, MultiheadAttention
from torch.nn.modules.transformer import TransformerEncoderLayer
from rlntm.modules.convNd import convNd
from typing import Optional, Any, List
import torch.nn.functional as F

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class HAttnEncoderLayer(Module):

    def __init__(self, input_size, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(HAttnEncoderLayer, self).__init__()
        
        self.nhead = nhead
        self.input_size = input_size

        self.attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.dropout = Dropout(dropout)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        if type(input_size) == int:
            input_size = [input_size]
        for size in input_size:
            # must be power of 2
            assert 2 ** int(math.log2(size)) == size
        self.convs = []
        size = list(input_size).copy()
        for i in range(int(math.log2(max(input_size)))):
            stdv = 1 / math.sqrt((d_model // nhead) * (2 ** len(size)))
            self.convs.append(
                    convNd(d_model // nhead, d_model // nhead, len(size), 2, stride=tuple(2 for _ in range(len(size))),
                        use_bias=True, padding=0, kernel_initializer=lambda x: torch.nn.init.uniform_(x, -stdv, stdv),
                        bias_initializer=lambda x: torch.nn.init.uniform_(x, -stdv, stdv)))
            for j in reversed(range(len(size))):
                size[j] = size[j] // 2
                if size[j] == 1:
                    del size[j]

        self.convs = nn.ModuleList(self.convs)

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
        out = src.permute(0, -1, *[k + 1 for k in range(len(src.size()) - 2)])
        hidden_vals = [out]
        for i in range(len(self.convs)):
            out = out.reshape(-1, src.size(-1) // self.nhead, *out.size()[2:])
            out = self.convs[i](out)
            out = self.activation(out)
            out = out.reshape(-1, src.size(-1), *out.size()[2:])

            out_exp = out
            for i in range(len(self.input_size)):
                out_exp = out_exp.repeat_interleave(hidden_vals[0].size(i + 2) // out_exp.size(i + 2), dim=i + 2)
            hidden_vals.append(out_exp)

        hidden_vals = torch.stack(hidden_vals, -1)
        hidden_vals = hidden_vals.transpose(1, -1).transpose(0,1).reshape(hidden_vals.size(-1), -1, hidden_vals.size(1))

        attn_out, _ = self.attn(hidden_vals[0:1], hidden_vals, hidden_vals)
        src2 = attn_out.reshape(src.size())

        src = src + self.dropout1(src2)
        src = self.norm1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class HAttnEncoder(Module):
    __constants__ = ['norm']

    def __init__(self, input_size, d_model, nhead, num_layers=32, dim_feedforward=512, dropout=0.1, activation="relu", layer_norm_eps = 1e-5):
        super(HAttnEncoder, self).__init__()
        self.layers = []
        for i in range(num_layers):
            self.layers.append(HAttnEncoderLayer(input_size, d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation="relu"))

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

