
import argparse
import parser

import numpy as np
import torch
import wandb
from tqdm import tqdm

import dataset
import longrange
import transformer
import util


def train_transformer(ds, net, n_batches=10, batch_size=32, seq_len=100, lr=1e-2, device='cpu', wandb=None, tqdm=None):
    net = net.to(device)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    
    pbar = dataset.get_random_batches(ds, n_batches, batch_size, seq_len)
    pbar = pbar if tqdm is None else tqdm(pbar, total=n_batches)
    for batch in pbar:
        batch = batch.to(device).long()

        loss = longrange.loss_fn(net, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if tqdm is not None:
            pbar.set_postfix(dict(loss=loss.item()))
        if wandb is not None:
            wandb.log(dict(loss=loss.item()))

def train_longrange(ds, net, n_batches=10, batch_size=32, seq_len=100, lr=1e-2, device='cpu', wandb=None, tqdm=None):
    net = net.to(device)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    pbar = dataset.get_random_skip_batches(ds, n_batches, batch_size, seq_len, min_dist=4*seq_len, max_dist=10*seq_len)
    pbar = pbar if tqdm is None else tqdm(pbar, total=n_batches)
    for batch1, batch2 in pbar:
        batch1, batch2 = batch1.to(device).long(), batch2.to(device).long()

        loss, loss1, loss2 = longrange.loss_fn_longrange(net, batch1, batch2)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if tqdm is not None:
            pbar.set_postfix(dict(loss=loss.item(), loss1=loss1.item(), loss2=loss2.item()))
        if wandb is not None:
            wandb.log(dict(loss=loss.item(), loss1=loss1.item(), loss2=loss2.item()))
            
# def train(ds, net, n_batches=10, batch_size=32, seq_len=100, lr=1e-2, device='cpu', wandb=None, tqdm=None):
#     net = net.to(device)
#     net.train()
#     opt = torch.optim.Adam(net.parameters(), lr=lr)

#     longrange = False
#     if longrange:
#         pbar = dataset.get_random_skip_batches(ds, n_batches, batch_size, seq_len, min_dist=4*seq_len, max_dist=10*seq_len)
#     else:
#         pbar = dataset.get_random_batches(ds, n_batches, batch_size, seq_len)
#     pbar = pbar if tqdm is None else tqdm(pbar, total=n_batches)
#     for batch in pbar:
#         # batch = batch.to(device).long()
#         # if longrange:
#             # batch1, batch2 = 
            

#         loss = longrange.loss_fn(net, batch)
#         opt.zero_grad()
#         loss.backward()
#         opt.step()

#         if tqdm is not None:
#             pbar.set_postfix(dict(loss=loss.item()))
#         if wandb is not None:
#             wandb.log(dict(loss=loss.item()))
#     for batch1, batch2 in pbar:
#         batch1, batch2 = batch1.to(device).long(), batch2.to(device).long()

#         loss, loss1, loss2 = longrange.loss_fn_longrange(net, batch1, batch2)
#         opt.zero_grad()
#         loss.backward()
#         opt.step()

#         if tqdm is not None:
#             pbar.set_postfix(dict(loss=loss.item(), loss1=loss1.item(), loss2=loss2.item()))
#         if wandb is not None:
#             wandb.log(dict(loss=loss.item(), loss1=loss1.item(), loss2=loss2.item()))
    

def main():
    args = parser.parse_args()
    print(args)

    print('Loading dataset...')
    ds_train, ds_test = dataset.load_dataset(tqdm=tqdm)

    print(args.model)
    np.random.seed(args.seed); torch.manual_seed(0)
    config_net = longrange.get_config(args.model)
    net = longrange.LongRangeGPT(**config_net)
    # if args.model == 'transformer':
    #     config = transformer.get_config('gpt-ak')
    #     net = transformer.GPT(config)
    #     train_fn = train_transformer
    # elif args.model == 'perceiver':
    #     config = transformer.get_config('gpt-ak', n_latent_tokens=args.n_latent_tokens)
    #     net = transformer.GPT(config)
    #     train_fn = train_transformer
    # elif args.model == 'longrange':
    #     config = longrange.get_config('gpt-ak')
    #     net = longrange.LongRangeGPT(config)
    #     train_fn = train_longrange
    print(f'Model config: {config_net}')
    print(f'Model # of parameters: {util.count_params(net)}')
    
    config = vars(args)
    config['config_net'] = config_net
    
    train_fn = train_longrange if config_net['use_memory'] else train_transformer
    
    if args.wandb:
        wandb.init(config=config)
    net.train()
    train_fn(ds_train, net, n_batches=args.n_batches, batch_size=args.batch_size,
             seq_len=args.seq_len, lr=args.lr, device=args.device, wandb=wandb if args.wandb else None, tqdm=tqdm)
    if args.wandb:
        net.eval()
        net = net.cpu()
        torch.save(net, f'../results/{wandb.run.name}.net')


parser = argparse.ArgumentParser(description='Train a model.')
parser.add_argument('--model', type=str, default='longrange')
parser.add_argument('--n_batches', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--seq_len', type=int, default=64)
parser.add_argument('--n_latent_tokens', type=int, default=None)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=False)


parser.add_argument('--lr', type=float, default=1e-3)

if __name__ == '__main__':
    main()

"""
python train.py --device cuda:0 --model transformer --seq_len 64 --seed 0

python train.py --device cuda:1 --model perceiver --seq_len 128 --seed 0
python train.py --device cuda:2 --model perceiver --seq_len 256 --seed 0
python train.py --device cuda:0 --model perceiver --seq_len 512 --seed 0

python train.py --device cuda:3 --model longrange1 --seq_len 64 --batch_size 64 --seed 0
"""
