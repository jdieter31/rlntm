import torch
from torch import nn
from .hdtransformer import HDTransformerEncoder, positionalencoding2d

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

class VisionTransformer(nn.Module):

    def __init__(self, num_classes=1000, d_model=1024, nhead=16, num_layers=15, dim_feedforward=4096, dropout=0.1, activation="relu", layer_norm_eps = 1e-5, chunk_size=16, normalize_dims=False, normal_dim=32):
        super(VisionTransformer, self).__init__()

        self.d_model = d_model
        self.hdtransformer = HDTransformerEncoder(2, d_model, nhead, num_layers, dim_feedforward, dropout, activation, layer_norm_eps)
        if normalize_dims:
            self.dim1normalizer = nn.MultiheadAttention(d_model, nhead, dropout, bias=True)
            self.dim2normalizer = nn.MultiheadAttention(d_model, nhead, dropout, bias=True)
        self.normalize_dims = normalize_dims
        self.normal_dim = normal_dim

        self.chunk_size = chunk_size
        self.embedder = nn.Linear(self.chunk_size * self.chunk_size * 3, d_model, bias=True)
        self.output_layer = nn.Linear(d_model, num_classes, bias=True)

        #print(get_n_params(self))

    def _chunk_tensor(self, tensor):
        chunks1 = torch.stack(torch.split(tensor, self.chunk_size, -2), -3)
        chunks2 = torch.stack(torch.split(chunks1, self.chunk_size, -1), -2)
        return chunks2.transpose(-2, -3).transpose(1, 2).transpose(2, 3).reshape(chunks2.size(0), chunks2.size(2), chunks2.size(4), -1)

    def forward(self, images, masks):
        chunked_images = self._chunk_tensor(images)
        embeddings = self.embedder(chunked_images)
        pos_encodings = positionalencoding2d(self.d_model, embeddings.size(-3), embeddings.size(-2))
        model_in = pos_encodings.unsqueeze(-1).transpose(0, -1).expand_as(embeddings).to(embeddings) + embeddings

        if self.normalize_dims:
            normal_pos_encoding = positionalencoding2d(self.d_model, model_in.size(1), self.normal_dim)
            decoder_in = normal_pos_encoding.unsqueeze(-1).transpose(0, -1)
            decoder_in = decoder_in.expand([model_in.size(0)] + list(decoder_in.size())[1:])

            dim1_input_model = model_in.transpose(0, 2).reshape(model_in.size(2), -1, model_in.size(-1))
            dim1_input_placeholder = decoder_in.transpose(0, 2).reshape(decoder_in.size(2), -1, decoder_in.size(-1)).to(dim1_input_model)
            normalized1, _ = self.dim1normalizer(dim1_input_placeholder, dim1_input_model, dim1_input_model)

            normalized1 = normalized1.reshape(decoder_in.transpose(0, 2).size()).transpose(0,2)

            normal_pos_encoding = positionalencoding2d(self.d_model, self.normal_dim, self.normal_dim)
            decoder_in = normal_pos_encoding.unsqueeze(-1).transpose(0, -1)
            decoder_in = decoder_in.expand([model_in.size(0)] + list(decoder_in.size())[1:])

            dim2_input_model = normalized1.transpose(0, 1).reshape(normalized1.size(1), -1, model_in.size(-1))
            dim2_input_placeholder = decoder_in.transpose(0, 2).reshape(decoder_in.size(2), -1, decoder_in.size(-1)).to(dim1_input_model)

            normalized2, _ = self.dim2normalizer(dim2_input_placeholder, dim2_input_model, dim2_input_model)

            normalized2 = normalized2.reshape(decoder_in.transpose(0, 2).size()).transpose(0,2)

            model_in = normalized2

        data = self.hdtransformer(model_in, src_key_padding_mask=masks)
        return self.output_layer(data.sum(-2).sum(-2) / (~masks).sum(-1).sum(-1).unsqueeze(-1))
