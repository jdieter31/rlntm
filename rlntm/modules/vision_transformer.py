import torch
from torch import nn
from .hdtransformer import HDTransformerEncoder

class VisionTransformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1, activation="relu", layer_norm_eps = 1e-5, chunk_size=16):
        super(VisionTransformer, self).__init__()

        self.hdtransformer = HDTransformerEncoder(2, d_model, nhead, num_layers, dim_feedforward, dropout, activation, layer_norm_eps)
        self.chunk_size = chunk_size

    def forward(self, images, masks):
        import ipdb; ipdb.set_trace()

