import torch
from torch import nn

from einops import repeat


def generate_batch(n_vocab, single_seqlen, n_seqs, bs, device=None):
    assert single_seqlen <= n_vocab
    return torch.stack([repeat(torch.randperm(n_vocab, device=device)[:single_seqlen], "n -> (l n)", l=n_seqs) for _ in range(bs)])

print(generate_batch(5, 5, 3, 100).shape)


# class MyRNN(nn.RNN):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def forward(self, input, h_0):
#         pass


# rnn = nn.RNN(input_size=32, hidden_size=32, num_layers=8, batch_first=True)

# x = torch.randn(16, 10, 32)

# h = None
# for i in range(10):
#     out, h = rnn(x[:, [i]], h)
#     print(out.shape, h.shape)
#     # x, h = rnn(x[:, i].unsqueeze(1), h)
