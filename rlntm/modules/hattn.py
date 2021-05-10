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

    def __init__(self, input_size=[512, 512], d_model=512, nhead=8, input_dim=3, num_layers=6, dim_feedforward=2048,
            dropout=0.1, activation="relu", layer_norm_eps = 1e-5, kernel_size=2,
            preprocessing_layers=5, output_head=True, output_size=1000):
        super(HAttnEncoder, self).__init__()

        if type(input_size) == int:
            input_size = [input_size]
        for size in input_size:
            # must be power of kernel_size
            assert kernel_size ** int(math.log2(size) / math.log2(kernel_size)) == size

        reduced_input_size = list(input_size).copy()

        for j in reversed(range(len(reduced_input_size))):
            reduced_input_size[j] = reduced_input_size[j] // (kernel_size ** preprocessing_layers)
            if reduced_input_size[j] == 1 or reduced_input_size[j] == 0:
                del reduced_input_size[j]


        self.layers = []
        for i in range(num_layers):
            self.layers.append(HAttnEncoderLayer(reduced_input_size, d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation="relu"))

        self.layers = nn.ModuleList(self.layers)
        self.convs = NDConvHierarchical(input_size, input_dim, d_model, nhead, preprocessing_layers, kernel_size)

        self.nhead = nhead
        self.num_layers = num_layers
        self.norm = LayerNorm(d_model, eps=layer_norm_eps)
        self.output_head = output_head
        if output_head:
            self.output_layer = nn.Linear(d_model, output_size, bias=True)


    def forward(self, src: Tensor) -> Tensor:

        # Do convolutions and copy values for nheads
        output = self.convs(src)
        for i, sliced in enumerate(output):
            sliced = sliced.unsqueeze(1).expand(-1, self.nhead, *sliced.size()[1:])
            sliced = sliced.reshape(sliced.size(0), -1, *sliced.size()[3:])
            output[i] = sliced

        for mod in self.layers:
            output = mod(output)


        if self.output_head:
            output = output[-1].reshape(output[-1].size(0), output[-1].size(1))
            if self.norm is not None:
                output = self.norm(output)
            return self.output_layer(output)

        return output

