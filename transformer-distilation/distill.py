import argparse
import os
from distutils.util import strtobool

import numpy as np
import torch
import wandb
from einops import rearrange
from torch import nn
from tqdm.auto import tqdm

import models
import train

parser = argparse.ArgumentParser()
parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--entity", type=str, default=None)
parser.add_argument("--project", type=str, default="transformer-distillation-toy")
parser.add_argument("--name", type=str, default=None)
# parser.add_argument("--log-video", type=lambda x: bool(strtobool(x)), default=False)
# parser.add_argument("--log-hist", type=lambda x: bool(strtobool(x)), default=False)

parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--seed", type=int, default=0)

parser.add_argument("--n-iters", type=lambda x: int(float(x)), default=int(1000))

parser.add_argument("--vocab-size", type=int, default=33)
parser.add_argument("--block-size", type=int, default=32)
parser.add_argument("--n-embd", type=int, default=64)
parser.add_argument("--n-layer", type=int, default=4)
parser.add_argument("--n-head", type=int, default=4)

parser.add_argument("--model", type=str, default="gpt")
parser.add_argument("--mode", type=str, default="random")

parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--dropout", type=float, default=0.0)

parser.add_argument("--save-model", type=str, default=None)


def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    if args.name:
        args.name = args.name.format(**vars(args))
    return args


def main(args):
    if args.track:
        wandb.init(entity=args.entity, project=args.project, name=args.name, config=args, save_code=True)
    # teacher_gpt = models.GPT(33, 32, 768, 4, 12)
    # teacher_gpt.load_state_dict(torch.load("models/gpt.pt"))
    # teacher_gpt = teacher_gpt.eval()
    teacher_comp = models.CompressionGPT(33, 32, 768, 4, 12, mode="random-causal")
    state_dict = torch.load("models/random-causal.pt")
    state_dict["mask"] = teacher_comp.mask
    teacher_comp.load_state_dict(state_dict)
    teacher_comp = teacher_comp.eval()

    teacher = teacher_comp
    teacher = teacher.to(args.device)

    # net = models.MyOneStepRecurrentNet(33, 768, 4 + 1)
    net = models.MyOneStepRecurrentTransformer(33, 768, 8, 12)
    net = net.to(args.device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    pbar = tqdm(range(args.n_iters))
    for i_iter in pbar:
        # -------------------------------------- TRAINING -------------------------------------- #
        tok = train.generate_batch(33, 33, args.batch_size, device=args.device)
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
        loss = 1.0 * loss1.mean() + 30 * loss2.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        # -------------------------------------- LOGGING -------------------------------------- #
        viz_slow = i_iter % np.clip(args.n_iters // 10, 1, None) == 0
        viz_midd = i_iter % np.clip(args.n_iters // 100, 1, None) == 0 or viz_slow
        viz_fast = i_iter % np.clip(args.n_iters // 1000, 1, None) == 0 or viz_midd
        data = {}
        if viz_fast:
            data["ppl"] = loss1.mean().exp().item()
            data["mse"] = loss2.mean().item()

        if viz_fast and args.track:
            wandb.log(data, step=i_iter)

        if args.save_model and viz_slow:
            os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
            torch.save(net.state_dict(), args.save_model)


if __name__ == "__main__":
    main(parse_args())
