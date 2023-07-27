import torch
from torch import nn
import numpy as np
from einops import repeat

import mygpt
from tqdm.auto import tqdm


vocab_size = 64
block_size = 32
n_embd = 768
n_layer = 4
n_head = 12


# def generate_batch(n_vocab, single_seqlen, n_seqs, bs, device=None):
# assert single_seqlen <= n_vocab
# return torch.stack([repeat(torch.randperm(n_vocab, device=device)[:single_seqlen], "n -> (l n)", l=n_seqs) for _ in range(bs)])
def generate_batch(n_vocab, single_seqlen, seqlen, bs, device=None):
    assert single_seqlen <= n_vocab
    n_seqs = seqlen // single_seqlen + 1
    return torch.stack([repeat(torch.randperm(n_vocab, device=device)[:single_seqlen], "n -> (l n)", l=n_seqs)[:seqlen] for _ in range(bs)])


bs = 16
device = "cpu"

net = mygpt.GPT(vocab_size, block_size, n_embd, n_layer, n_head, dropout=0.0, bias=True)
net = net.to(device)

opt = torch.optim.Adam(net.parameters(), lr=1e-4)

n_iters = 100
pbar = tqdm(range(n_iters))
for i_iter in pbar:
    if i_iter == n_iters - 1:
        bs = 512
    tok = generate_batch(vocab_size, 30, block_size + 1, bs, device=device)
    x, y = tok[:, :-1], tok[:, 1:]
    logits = net(x)

    loss_pred = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), reduction="none").reshape(bs, -1)
    loss = loss_pred.mean()
    opt.zero_grad()
    loss.backward()
    opt.step()

    pbar.set_postfix(loss=loss.item(), ppl=loss.exp().item(), ppl0=loss_pred[:, 0].mean().exp().item(), ppl1=loss_pred[:, -1].mean().exp().item())

import matplotlib.pyplot as plt

print(tok[0].cpu().numpy())
print(tok[1].cpu().numpy())
print(tok[2].cpu().numpy())
print(tok[3].cpu().numpy())

plt.plot(loss_pred.detach().cpu().mean(dim=0).exp().numpy())
plt.show()
