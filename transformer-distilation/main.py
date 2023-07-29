import argparse
from distutils.util import strtobool

import numpy as np
import torch
from einops import repeat
from torch import nn
from tqdm.auto import tqdm

import models
import wandb
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--entity", type=str, default=None)
parser.add_argument("--project", type=str, default="transformer-distillation-toy")
parser.add_argument("--name", type=str, default=None)
# parser.add_argument("--log-video", type=lambda x: bool(strtobool(x)), default=False)
# parser.add_argument("--log-hist", type=lambda x: bool(strtobool(x)), default=False)

parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--seed", type=int, default=0)

parser.add_argument("--n-iters", type=lambda x: int(float(x)), default=int(5000))

parser.add_argument("--vocab-size", type=int, default=32)
# convert these vars to argparse args
parser.add_argument("--block-size", type=int, default=32)
parser.add_argument("--n-embd", type=int, default=64)
parser.add_argument("--n-layer", type=int, default=4)
parser.add_argument("--n-head", type=int, default=4)

parser.add_argument("--arch", type=str, default="causal")

parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--dropout", type=float, default=0.0)


def generate_batch(n_vocab, single_seqlen, seqlen, bs, device=None):
    assert single_seqlen <= n_vocab
    n_seqs = seqlen // single_seqlen + 1
    return torch.stack([repeat(torch.randperm(n_vocab, device=device)[:single_seqlen], "n -> (l n)", l=n_seqs)[:seqlen] for _ in range(bs)])


def main(args):
    if args.track:
        wandb.init(entity=args.entity, project=args.project, name=args.name, config=args, save_code=True)

    # if args.arch == "rnn":
    # net = models.MyRNN(args.vocab_size, args.n_embd, args.n_embd, args.n_layer, nonlinearity="relu", batch_first=True, dropout=args.dropout)
    # else:
    # net = models.GPT(args.vocab_size, args.block_size, args.n_embd, args.n_layer, args.n_head, dropout=args.dropout, bias=True)
    net = models.Compression2AK(args.vocab_size, args.block_size, args.n_embd, args.n_layer, args.n_head, dropout=args.dropout, bias=True)

    net = net.to(args.device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    pbar = tqdm(range(args.n_iters))
    for i_iter in pbar:
        viz_slow = i_iter % np.clip(args.n_iters // 10, 1, None) == 0
        viz_midd = i_iter % np.clip(args.n_iters // 100, 1, None) == 0 or viz_slow
        viz_fast = i_iter % np.clip(args.n_iters // 1000, 1, None) == 0 or viz_midd
        batch_size = 1024 if viz_slow else args.batch_size
        # -------------------------------------- TRAINING -------------------------------------- #
        tok = generate_batch(args.vocab_size, args.vocab_size, args.block_size + 1, batch_size, device=args.device)
        x, y = tok[:, :-1], tok[:, 1:]

        loss_pred = net.loss_fn(tok)
        loss = loss_pred.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        # -------------------------------------- LOGGING -------------------------------------- #
        data = {}
        if viz_fast:
            data["loss"] = loss.item()
            data["ppl"] = loss.exp().item()
        if viz_slow:
            ppl = loss_pred.detach().cpu().nanmean(dim=0).exp().numpy()
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
            plt.close("all")
        # pbar.set_postfix(loss=loss.item(), ppl=loss.exp().item(), ppl0=loss_pred[:, 0].mean().exp().item(), ppl1=loss_pred[:, -1].mean().exp().item())


if __name__ == "__main__":
    main(parser.parse_args())
