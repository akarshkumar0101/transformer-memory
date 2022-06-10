import argparse

import torch
from transformers import AutoTokenizer

from modeling_gpt2 import GPT2Model, GPT2LMHeadModel

# from hopfield_memory import HopfieldMemory
import perplexity


parser = argparse.ArgumentParser()

parser.add_argument("--wandb", action="store_true")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--hm", action="store_true")
parser.add_argument("--beta1", type=float, default=1.0)
parser.add_argument("--beta2", type=float, default=1.0)
parser.add_argument("--beta3", type=float, default=1.0)
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--adaptive_alpha", action="store_true")
parser.add_argument("--opt", type=str, default='sgd')
parser.add_argument("--lr", type=float, default=1.)


def main(args=None):
    args = parser.parse_args(args=args)
    print(args)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.wandb:
        import wandb
        wandb.init(config=args)
    else:
        wandb = None
    print(wandb.config)

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")

    print(f"Using device {device}")

    perplexity.calc_perplexity(
        tokenizer,
        model,
        text=None,
        context_length=100,
        stride=1,
        device=device,
        wandb=wandb,
    )

    wandb.finish()

if __name__ == "__main__":
    main()
