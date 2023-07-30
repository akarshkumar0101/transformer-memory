import argparse
from distutils.util import strtobool

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from einops import rearrange, repeat
from torch import nn
from tqdm.auto import tqdm

import models

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

parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--dropout", type=float, default=0.0)


def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    if args.name:
        args.name = args.name.format(**vars(args))
    return args


# def generate_batch(vocab_size, single_seqlen, seqlen, bs, device=None):
#     assert single_seqlen <= vocab_size
#     n_seqs = seqlen // single_seqlen + 1
#     return torch.stack([repeat(torch.randperm(vocab_size, device=device)[:single_seqlen], "n -> (l n)", l=n_seqs)[:seqlen] for _ in range(bs)])


def generate_batch(vocab_size, ctx_len, batch_size, device=None):
    assert ctx_len <= vocab_size
    return torch.stack([torch.randperm(vocab_size, device=device)[:ctx_len] for _ in range(batch_size)])


def main(args):
    if args.track:
        wandb.init(entity=args.entity, project=args.project, name=args.name, config=args, save_code=True)

    if args.model == "gpt":
        net = models.GPT(args.vocab_size, args.block_size, args.n_embd, args.n_layer, args.n_head, dropout=args.dropout, bias=True)
    elif args.model == "compress-gpt":
        net = models.CompressionGPT(args.vocab_size, args.block_size, args.n_embd, args.n_layer, args.n_head, dropout=args.dropout, bias=True, mode=args.mode)
    elif args.model == "weird":
        net = models.WeirdGPT(args.vocab_size, args.block_size, args.n_embd, args.n_layer, args.n_head, dropout=args.dropout, bias=True)
    elif args.model == "rnn":
        net = models.MyRNN(args.vocab_size, args.n_embd, args.n_embd, args.n_layer, nonlinearity="relu", batch_first=True, dropout=args.dropout)
    else:
        raise NotImplementedError

    net = net.to(args.device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    pbar = tqdm(range(args.n_iters))
    for i_iter in pbar:
        viz_slow = i_iter % np.clip(args.n_iters // 10, 1, None) == 0
        viz_midd = i_iter % np.clip(args.n_iters // 100, 1, None) == 0 or viz_slow
        viz_fast = i_iter % np.clip(args.n_iters // 1000, 1, None) == 0 or viz_midd
        batch_size = 1024 if viz_slow else args.batch_size
        # -------------------------------------- TRAINING -------------------------------------- #
        tok = generate_batch(args.vocab_size, args.block_size + 1, batch_size, device=args.device)
        x, y = tok[:, :-1], tok[:, 1:]
        logits = net(x)

        loss_pred = torch.nn.functional.cross_entropy(rearrange(logits, "b t d -> (b t) d"), rearrange(y, "b t -> (b t)"), reduction="none")
        loss_pred = rearrange(loss_pred, "(b t) -> b t", b=batch_size)
        # loss_pred = torch.nn.functional.cross_entropy(logits.reshape(-1, args.vocab_size), y.reshape(-1), reduction="none").reshape(batch_size, -1)

        loss = loss_pred.nanmean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        # -------------------------------------- LOGGING -------------------------------------- #
        loss, loss_pred = loss.detach().cpu(), loss_pred.detach().cpu()
        data = {}
        if viz_fast:
            data["loss"] = loss.item()
            data["ppl"] = loss.exp().item()
            data["ppl_init"] = loss_pred[:, 0].nanmean().exp().item()
            data["ppl_half"] = loss_pred[:, args.block_size // 2].nanmean().exp().item()
            data["ppl_final"] = loss_pred[:, -1].nanmean().exp().item()
        if viz_slow:
            ppl = loss_pred.nanmean(dim=0).exp().numpy()
            pos = np.arange(len(ppl))

            # plt.plot(pos, ppl)
            # plt.ylim(0, args.vocab_size)
            # plt.ylabel("PPL")
            # plt.xlabel("Token Position")
            # data["mpl"] = plt
            table = wandb.Table(data=np.stack([pos, ppl], axis=-1), columns=["pos", "ppl"])
            if args.track:
                data["manual"] = wandb.plot.line(table, "pos", "ppl", title="PPL vs Position")

        if args.track and viz_fast:
            wandb.log(data, step=i_iter)
            # plt.close("all")
        if viz_fast:
            pbar.set_postfix(ppl=data["ppl"], ppl_init=data["ppl_init"], ppl_half=data["ppl_half"], ppl_final=data["ppl_final"])


if __name__ == "__main__":
    main(parse_args())
