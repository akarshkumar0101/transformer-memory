import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]
    
def calc_fourier_position_embeddings(pos, config):
    dmodel = config["n_embd"]
    # pos.shape is (1, context_length)
    pe = torch.zeros(*pos.shape, dmodel)
    pos = pos[..., None]
    i2divdmodel = torch.arange(0, dmodel, 2) / dmodel
    pe[..., ::2] = torch.sin(pos / 10000**i2divdmodel)
    pe[..., 1::2] = torch.cos(pos / 10000**i2divdmodel)
    return pe

