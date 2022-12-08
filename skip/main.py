import argparse
from functools import partial

import dataset
import longrange
import numpy as np
import torch
import train
import util
import wandb
from tqdm.auto import tqdm


def main():
    args = parser.parse_args()
    print(args)
    print('Loading dataset...')
    ds_train, ds_test = dataset.load_dataset(tqdm=tqdm)
    print('Done!')
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    config_net = longrange.get_config(args.model)
    net = longrange.LongRangeGPT(**config_net)
    print(f'Model: {args.model}')
    print(f'Model config: {config_net}')
    print(f'Model # of parameters: {util.count_params(net)}')
    
    config = vars(args)
    config['config_net'] = config_net
    
    if config_net['use_memory']:
        train_fn = partial(train.train_longrange, min_dist=args.min_dist, max_dist=args.max_dist)
    else:
        train_fn = train.train_transformer
    
    if args.track:
        wandb.init(config=config, name=args.name, save_code=True)

    net.train()
    train_fn(ds_train, net, n_batches=args.n_batches, batch_size=args.batch_size,
             seq_len=args.seq_len, lr=args.lr, device=args.device, wandb=wandb if args.track else None, tqdm=tqdm)

    if args.track:
        net.eval().cpu()
        torch.save(net, f'../results/{wandb.run.name}.net')

parser = argparse.ArgumentParser(description='Train a model.')
parser.add_argument('--track', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--name', type=str, default=None)

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=str, default='cpu')

parser.add_argument('--model', type=str, default='longrange')
parser.add_argument('--n_batches', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--seq_len', type=int, default=64)
parser.add_argument('--min_dist', type=int, default=0)
parser.add_argument('--max_dist', type=int, default=1)
parser.add_argument('--n_latent_tokens', type=int, default=None)

parser.add_argument('--lr', type=float, default=1e-3)

if __name__ == '__main__':
    main()

"""
python train.py --device cuda:0 --model transformer --seq_len 64 --seed 0

python train.py --device cuda:1 --model perceiver --seq_len 128 --seed 0
python train.py --device cuda:2 --model perceiver --seq_len 256 --seed 0
python train.py --device cuda:0 --model perceiver --seq_len 512 --seed 0

python train.py --device cuda:3 --model longrange1 --seq_len 64 --batch_size 64 --seed 0


python main.py --device cuda:0 --model transformer --seq_len 002 --track --name transformer-002
python main.py --device cuda:0 --model transformer --seq_len 004 --track --name transformer-004
python main.py --device cuda:0 --model transformer --seq_len 008 --track --name transformer-008
python main.py --device cuda:0 --model transformer --seq_len 016 --track --name transformer-016

python main.py --device cuda:1 --model transformer --seq_len 032 --track --name transformer-032
python main.py --device cuda:1 --model transformer --seq_len 064 --track --name transformer-064

python main.py --device cuda:2 --model  longrange1 --seq_len 002 --track --name  longrange1-002
python main.py --device cuda:2 --model  longrange1 --seq_len 004 --track --name  longrange1-004
python main.py --device cuda:2 --model  longrange1 --seq_len 008 --track --name  longrange1-008
python main.py --device cuda:2 --model  longrange1 --seq_len 016 --track --name  longrange1-016

python main.py --device cuda:3 --model  longrange1 --seq_len 032 --track --name  longrange1-032
python main.py --device cuda:3 --model  longrange1 --seq_len 064 --track --name  longrange1-064

"""