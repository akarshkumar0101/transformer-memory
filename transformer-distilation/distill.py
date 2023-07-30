import torch
from einops import rearrange
from torch import nn
from tqdm.auto import tqdm

import train
import models

device = 'mps'

teacher_gpt = models.GPT(33, 32, 768, 4, 12)
teacher_gpt.load_state_dict(torch.load("models/gpt.pt"))
teacher_gpt = teacher_gpt.eval()

teacher_comp = models.CompressionGPT(33, 32, 768, 4, 12, mode="random-causal")
state_dict = torch.load("models/random-causal.pt")
state_dict["mask"] = teacher_comp.mask
teacher_comp.load_state_dict(state_dict)
teacher_comp = teacher_comp.eval()

teacher = teacher_comp
teacher = teacher.to(device)

def main():
    # net = models.MyOneStepRecurrentNet(33, 768, 4 + 1)
    net = models.MyOneStepRecurrentTransformer(33, 768, 6, 12)
    net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)

    data = []
    for i_iter in tqdm(range(10000)):
        bs = 128 if i_iter % 200 == 0 else 16
        tok = train.generate_batch(33, 33, bs, device='cpu').to(device)
        x, y = tok[:, :-1], tok[:, 1:]

        with torch.no_grad():
            activations = teacher(x, mode="encode-causal")
            activations = activations

        mem1 = rearrange(activations[:, :, :-2, :], "b l t d -> (b t) l d")  # red
        mem2 = rearrange(activations[:, :, 1:-1, :], "b l t d -> (b t) l d")  # blue
        mem3 = rearrange(activations[:, :, 2:, :], "b l t d -> (b t) l d")  # orange
        tok1 = rearrange(x[:, :-2], "b t -> (b t)")  # red
        tok2 = rearrange(x[:, 1:-1], "b t -> (b t)")  # blue
        tok3 = rearrange(x[:, 2:], "b t -> (b t)")  # orange

        logits, mem2hat = net(tok2, mem1)
        loss1 = nn.functional.cross_entropy(logits, tok3, reduction="none")
        loss2 = (mem2hat - mem2).pow(2)
        loss = 1.0 * loss1.mean() + 20 * loss2.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        data.append((loss1.mean().exp().item(), loss2.mean().item()))
        if i_iter % 200 == 0:
            print(f"i: {i_iter: 8d} ppl: {loss1.mean().exp().item():8.3f}, mse: {loss2.mean().item():8.3f}")


if __name__ == "__main__":
    main()
