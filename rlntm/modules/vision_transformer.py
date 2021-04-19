import torch
from torch import nn
from .hdtransformer import HDTransformerEncoder, positionalencoding2d

class VisionTransformer(nn.Module):

    def __init__(self, num_classes=1000, d_model=32, nhead=8, num_layers=6, dim_feedforward=128, dropout=0, activation="relu", layer_norm_eps = 1e-5, chunk_size=16):
        super(VisionTransformer, self).__init__()

        self.d_model = d_model
        self.hdtransformer = HDTransformerEncoder(2, d_model, nhead, num_layers, dim_feedforward, dropout, activation, layer_norm_eps)
        self.chunk_size = chunk_size
        self.embedder = nn.Linear(self.chunk_size * self.chunk_size * 3, d_model, bias=True)
        self.output_layer = nn.Linear(d_model, num_classes, bias=True)

    def _chunk_tensor(self, tensor):
        chunks1 = torch.stack(torch.split(tensor, self.chunk_size, -2), -3)
        chunks2 = torch.stack(torch.split(chunks1, self.chunk_size, -1), -2)
        return chunks2.transpose(-2, -3).transpose(1, 2).transpose(2, 3).reshape(chunks2.size(0), chunks2.size(2), chunks2.size(4), -1)

    def forward(self, images, masks):
        chunked_images = self._chunk_tensor(images)
        embeddings = self.embedder(chunked_images)
        pos_encodings = positionalencoding2d(self.d_model, embeddings.size(-3), embeddings.size(-2))
        model_in = pos_encodings.unsqueeze(-1).transpose(0, -1).expand_as(embeddings).to(embeddings) + embeddings
        data = self.hdtransformer(model_in, src_key_padding_mask=masks)
        with torch.no_grad():
            data[data != data] = 0
        return self.output_layer(data.mean(-2).mean(-2))
