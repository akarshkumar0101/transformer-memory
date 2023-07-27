import model
import mygpt
import torch
from torch import nn


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



mha = nn.MultiheadAttention(embed_dim=768, num_heads=12, dropout=0.0, batch_first=True)

x = torch.randn(16, 32, 768)

mask = torch.triu(torch.ones(32, 32, dtype=torch.bool), diagonal=1)
out, attn = mha(x, x, x, attn_mask=mask, need_weights=True, average_attn_weights=False)

a = attn[0, 0]
print(a.sum(dim=-1))
import matplotlib.pyplot as plt
plt.imshow(a.detach().cpu().numpy())
plt.colorbar()
plt.show()



