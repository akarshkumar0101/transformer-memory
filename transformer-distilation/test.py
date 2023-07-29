import model
import models
import torch
from torch import nn

import matplotlib.pyplot as plt

# config = model.GPTConfig(block_size=32, vocab_size=0, n_layer=4, n_head=12, n_embd=768, dropout=0.1, bias=True)
# net = model.CausalSelfAttention(config)
# print(sum(p.numel() for p in net.parameters()))
# for name, p in net.named_parameters():
#     print(name, p.shape)

# net = nn.MultiheadAttention(embed_dim=768, num_heads=12, dropout=0.1)
# print(sum(p.numel() for p in net.parameters()))
# for name, p in net.named_parameters():
#     print(name, p.shape)


# print('-------')

# net = model.Block(config)
# print(sum(p.numel() for p in net.parameters()))
# x = torch.randn(32, 16, 768)
# print(net(x).shape)
# print(net.state_dict().keys())

# net = mygpt.Block(n_embd=768, n_head=12, dropout=0.1, bias=True)
# print(sum(p.numel() for p in net.parameters()))
# print(net.state_dict().keys())


# mha = nn.MultiheadAttention(embed_dim=768, num_heads=12, dropout=0.0, batch_first=True)

# x = torch.randn(16, 32, 768)

# mask = torch.triu(torch.ones(32, 32, dtype=torch.bool), diagonal=1)
# out, attn = mha(x, x, x, attn_mask=mask, need_weights=True, average_attn_weights=False)

# a = attn[0, 0]
# print(a.sum(dim=-1))
# plt.imshow(a.detach().cpu().numpy())
# plt.colorbar()
# plt.show()


# def create_causal_mask():
#     mask = torch.tril(torch.ones(32, 32, dtype=torch.bool))
#     return mask


# def create_weird_mask():
#     mask = torch.tril(torch.ones(32, 32, dtype=torch.bool))
#     half = 32//2
#     mask[half:, :half-1] = False
#     mask[:half, :half] = True
#     return mask


# mask = create_causal_mask()
# plt.subplot(121)
# plt.imshow(mask)
# mask = create_weird_mask()
# plt.subplot(122)
# plt.imshow(mask)
# plt.show()


# import wandb
# import numpy as np

# wandb.init(project="test")

# for i in range(10):
#     x = np.arange(10) + i
#     plt.plot(x)
#     plt.ylabel("some interesting numbers")
#     wandb.log({"chart": plt})
#     plt.close("all")


# net = mygpt.GPT(32, 32, 768, 4, 12, arch='weird')

# x = torch.randint(0, 32, (16, 32))
# y = net(x)

# for i in range(4):
#     plt.imshow(net.h[i].attn_weights[3, 7].detach().cpu().numpy())
#     plt.show()



rnn = nn.RNN(32, 32, 5, batch_first=True)
rnn = nn.Sequential(nn.Embedding(32, 32), rnn)


x = torch.randint(0, 32, (16, 10))
out, h = rnn(x)
print(out.shape)

# assert torch.allclose(out[:, -1, :], h[-1])