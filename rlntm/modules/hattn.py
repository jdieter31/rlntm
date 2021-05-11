import torch.nn as nn
import math
import torch
from torch import Tensor
from torch.nn import Module, ModuleList, LayerNorm, Linear, Dropout, MultiheadAttention
from torch.nn.modules.transformer import TransformerEncoderLayer
from rlntm.modules.convNd import convNd
from typing import Optional, Any, List
import torch.nn.functional as F
import numpy as np

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def nd_mean_pool(tensor, n, kernel_size=2):
    # tensor size [..., E, d_1, ..., d_n]
    for i in range(n):
        tensor = torch.stack(torch.split(tensor, kernel_size, - 1 - i), - 2 - i).mean(-1 - i)
    return tensor

def chunk_tensor(tensor, n, kernel_size=2):
    # tensor size [..., E, d_1, ..., d_n]

    for i in range(n):
        splits = torch.split(tensor, kernel_size, - 1 - 2 * i)
        tensor = torch.stack(splits, - 2 - 2 * i)

    n_dims = (len(tensor.size()) - 2) // 2
    tensor = tensor.permute(0, 1, *[2 * i + 2 for i in range(n_dims)], *[2 * i + 3 for i in range(n_dims)])
    return tensor

def unchunk_tensor(tensor, n, kernel_size=2):

    n_dims = (len(tensor.size()) - 2) // 2
    tensor = tensor.permute(0, 1, *[i // 2 + (i % 2) * n_dims + 2 for i in range(2 * n_dims)])
    tensor = tensor.reshape(tensor.size(0), tensor.size(1), *[tensor.size(2 + 2 * i) * kernel_size for i in range(n_dims)])

    return tensor

class HAttnEncoderLayer(Module):

    def __init__(self, input_size, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", kernel_size=2):
        super(HAttnEncoderLayer, self).__init__()

        self.nhead = nhead

        if type(input_size) == int:
            input_size = [input_size]
        for size in input_size:
            # must be power of kernel_size
            assert kernel_size ** int(math.log2(size) / math.log2(kernel_size)) == size

        self.kernel_size = kernel_size
        self.input_size = input_size

        seq_length = int(math.log2(max(input_size)) / math.log2(kernel_size))
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation) for _ in range(seq_length)])

    def forward(self, src: Tensor) -> Tensor:

        for i in range(len(src) - 1):
            vec = 0
            chunks = chunk_tensor(src[i], len(self.input_size), self.kernel_size)
            chunks = chunks.reshape(*chunks.size()[:2 + len(self.input_size)], -1)

            if i == 0:
                attn_in = torch.cat([chunks, src[i + 1].unsqueeze(-1),
                    src[-1].expand_as(src[i + 1]).unsqueeze(-1)], dim=-1)
            else:
                attn_in = torch.cat([chunks, src[i + 1].unsqueeze(-1)], dim=-1)

            attn_in_1 = attn_in.permute(-1, 0, *[2 + i for i in range(len(self.input_size))], 1)
            attn_in_2 = attn_in_1.reshape(attn_in_1.size(0), -1, attn_in_1.size(-1))

            out = self.layers[i](attn_in_2)
            out = out.reshape(attn_in_1.size())
            out = out.permute(1, -1, *[2 + i for i in range(len(self.input_size))], 0)

            new_chunks = out.transpose(0,-1)[0:self.kernel_size ** len(self.input_size)].transpose(0,-1)
            new_chunks = new_chunks.reshape(chunks.size())
            new_chunks = new_chunks.reshape(*chunks.size()[0:-1], *(len(self.input_size) * [self.kernel_size]))

            src[i] = unchunk_tensor(new_chunks, len(self.input_size), self.kernel_size)

            src[i + 1] = out.transpose(0,-1)[
                    self.kernel_size ** len(self.input_size):
                    (self.kernel_size ** len(self.input_size))+1
                    ].transpose(0,-1).squeeze(dim=-1)

        return src

class NDConvHierarchical(Module):

    def __init__(self, input_size, input_channels, d_model, nhead, preprocessing_layers=5, kernel_size=2, activation="relu"):
        super(NDConvHierarchical, self).__init__()

        channels = d_model // nhead

        self.preprocessing_convs = []
        self.input_size = input_size
        self.kernel_size = kernel_size

        size = list(input_size).copy()

        stdv = 1 / math.sqrt((max(channels // (kernel_size ** (preprocessing_layers-1)), input_channels)) * (kernel_size ** len(size)))
        self.preprocessing_convs.append(
                convNd(input_channels, max(channels // (kernel_size ** (preprocessing_layers - 1)), input_channels),
                    len(size), 3, stride=1,
                    use_bias=True, padding=1, kernel_initializer=lambda x: torch.nn.init.uniform_(x, -stdv, stdv),
                    bias_initializer=lambda x: torch.nn.init.uniform_(x, -stdv, stdv)))

        for j in reversed(range(len(size))):
            size[j] = size[j] // kernel_size
            if size[j] == 1:
                del size[j]

        for i in reversed(range(preprocessing_layers - 1)):
            stdv = 1 / math.sqrt((max(channels // (kernel_size ** (i)), input_channels)) * (kernel_size ** len(size)))
            self.preprocessing_convs.append(
                    convNd(max(channels // (kernel_size ** (i + 1)), input_channels),  max(channels // (kernel_size ** (i)), input_channels),
                        len(size), 3, stride=1,
                        use_bias=True, padding=1, kernel_initializer=lambda x: torch.nn.init.uniform_(x, -stdv, stdv),
                        bias_initializer=lambda x: torch.nn.init.uniform_(x, -stdv, stdv)))

            for j in reversed(range(len(size))):
                size[j] = size[j] // kernel_size
                if size[j] == 1:
                    del size[j]

        self.preprocessing_convs = nn.ModuleList(self.preprocessing_convs)

        self.convs = []
        for i in range(int(math.log(max(input_size)) / math.log(kernel_size)) - preprocessing_layers):
            stdv = 1 / math.sqrt((d_model // nhead) * (kernel_size ** len(size)))
            self.convs.append(
                    convNd(d_model // nhead, d_model // nhead, len(size), 3, stride=1,
                        use_bias=True, padding=1, kernel_initializer=lambda x: torch.nn.init.uniform_(x, -stdv, stdv),
                        bias_initializer=lambda x: torch.nn.init.uniform_(x, -stdv, stdv)))
            for j in reversed(range(len(size))):
                size[j] = size[j] // kernel_size
                if size[j] == 1:
                    del size[j]

        self.convs = nn.ModuleList(self.convs)

        self.activation = _get_activation_fn(activation)

    def forward(self, src):
        out = src
        for i in range(len(self.preprocessing_convs)):
            out = self.preprocessing_convs[i](out)
            out = self.activation(out)
            out = nd_mean_pool(out, len(self.input_size), self.kernel_size)

        saved_vals = [out]
        for i in range(len(self.convs)):
            out = self.convs[i](out)
            out = self.activation(out)
            out = nd_mean_pool(out, len(self.input_size), self.kernel_size)

            saved_vals.append(out)

        return saved_vals


class HAttnEncoder(Module):
    __constants__ = ['norm']

    def __init__(self, input_size=[256, 256], d_model=512, nhead=8, input_dim=3, num_layers=8, dim_feedforward=2048,
            dropout=0.1, activation="relu", layer_norm_eps = 1e-5, kernel_size=2,
            output_head=True, output_size=1000, pos_encodings=False, chunk_size=16):
        super(HAttnEncoder, self).__init__()

        if type(input_size) == int:
            input_size = [input_size]
        self.input_size = input_size
        reduced_input_size = list(input_size).copy()
        self.pos_encodings = pos_encodings
        self.d_model = d_model


        """
        for j in reversed(range(len(reduced_input_size))):
            reduced_input_size[j] = reduced_input_size[j] // (kernel_size ** preprocessing_layers)
            if reduced_input_size[j] == 1 or reduced_input_size[j] == 0:
                del reduced_input_size[j]
        """
        self.reduced_input_size = [size // chunk_size for size in input_size]


        self.layers = []
        for i in range(num_layers):
            self.layers.append(HAttnEncoderLayer(self.reduced_input_size, d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation="relu"))

        self.layers = nn.ModuleList(self.layers)
        #self.convs = NDConvHierarchical(input_size, input_dim, d_model, nhead, preprocessing_layers, kernel_size)
        self.chunk_size = chunk_size
        self.embedder = nn.Linear(input_dim * (chunk_size ** len(input_size)), d_model)
        self.nhead = nhead
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.norm = LayerNorm(d_model, eps=layer_norm_eps)
        self.output_head = output_head
        if output_head:
            self.output_layer = nn.Linear(d_model, output_size, bias=True)


    def forward(self, src: Tensor) -> Tensor:

        # Do convolutions and copy values for nheads
        # output = self.convs(src)
        # for i, sliced in enumerate(output):
        #    sliced = sliced.unsqueeze(1).expand(-1, self.nhead, *sliced.size()[1:])
        #    sliced = sliced.reshape(sliced.size(0), -1, *sliced.size()[3:])
        #    output[i] = sliced

        src = chunk_tensor(src, len(self.input_size), self.chunk_size)
        src = src.permute(0, *[i + 2 for i in range(len(src.size()) - 2)], 1)
        src = src.reshape(*src.size()[:- len(self.input_size) - 1], -1)
        src = self.embedder(src)
        src = src.permute(0, -1, *[i + 1 for i in range(len(src.size()) - 2)])


        seq_length = int(math.log2(max(self.reduced_input_size)) / math.log2(self.kernel_size)) + 1

        output = []
        for i in range(seq_length):
            output.append(src)

            if self.pos_encodings:
                output[i] = output[i] + positionalencoding2d(self.d_model, output[i].size(-2), output[i].size(-1)).unsqueeze(0).to(output[i])

            src = nd_mean_pool(src, len(self.input_size), self.kernel_size)

        for mod in self.layers:
            output = mod(output)

        if self.output_head:
            output = output[-1].reshape(output[-1].size(0), output[-1].size(1))
            if self.norm is not None:
                output = self.norm(output)
            return self.output_layer(output)

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
