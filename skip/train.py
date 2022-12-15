
import argparse
import os
import parser
from functools import partial

import dataset
import evaluate
import longrange
import lovely_tensors as lt
import matplotlib.pyplot as plt
import numpy as np
import torch
import util
import wandb
from tqdm import tqdm
from tqdm.auto import tqdm

lt.monkey_patch()



def train_transformer(ds, net, n_batches=10, batch_size=32, seq_len=100, lr=1e-3, device='cpu', wandb=None, tqdm=None):
    net.to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    
    pbar = dataset.get_seq_batches(ds, n_batches, batch_size, n_seqs=1, seq_len=seq_len, min_dist=0, max_dist=1, unbind=True)
    pbar = pbar if tqdm is None else tqdm(pbar, total=n_batches)
    for (batch1,), (batch1fchar,) in pbar:
        batch1 = batch1.to(device).long()

        loss = longrange.loss_fn(net, batch1)
        opt.zero_grad()
        loss.backward()
        opt.step()

        grad_main = torch.cat([p.grad.flatten() for p in net.blocks_main.parameters()]).norm().item()

        data = dict(loss=loss.item(), grad_main=grad_main)
        if tqdm is not None:
            pbar.set_postfix({k: v for k, v in data.items() if isinstance(v, float)})
        if wandb is not None:
            wandb.log(data)

def train_longrange(net, ds, n_batches, batch_size,
                    n_seqs, seq_len, min_dist, max_dist,
                    lr=1e-3, device='cpu', wandb=None, tqdm=None):
    net.to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    # scheduler = torch.optim.MultiStepLR(opt, milestones=[1000], gamma=0.1)

    pbar = dataset.get_seq_batches(ds, n_batches, batch_size, n_seqs, seq_len,
                                   min_dist, max_dist, unbind=False, device=device)
    pbar = pbar if tqdm is None else tqdm(pbar, total=n_batches)
    for ids, fbins in pbar:
        # ids: (bs, n_seqs, seq_len), fbins: (bs, n_seqs, seq_len)
        losses = longrange.loss_fn_longrange(net, ids) #: (bs, n_seqs, seq_len-1)
        loss = losses.mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        grad_main = torch.cat([p.grad.flatten() for p in net.blocks_main.parameters()]).norm()
        
        data = dict(ppl=loss.exp().item())
        data.update({f'ppl_seq_{seq_i}': losses_seq_i.mean().exp().item()
                     for seq_i, losses_seq_i in enumerate(losses.unbind(dim=-2))})
        data.update(grad_main=grad_main.item())
        if net.config['use_memory']:
            grad_mc = torch.cat([p.grad.flatten() for p in net.blocks_memory_creator.parameters()]).norm()
            grad_mr = torch.cat([p.grad.flatten() for p in net.blocks_memory_retriever.parameters()]).norm()
            data.update(grad_mc=grad_mc.item(), grad_mr=grad_mr.item())

        if tqdm is not None:
            pbar.set_postfix({k: v for k, v in data.items() if isinstance(v, float)})
        if wandb is not None:
            wandb.log(data)
            
# def log_network_stats(akl, net):
#     grad_mag = [block.attn.out_proj.weight.grad.norm().item() for block in net.blocks_main]
#     akl.put(grad_blocks_main=grad_mag)
#     grad_mag = [block.attn.out_proj.weight.grad.norm().item() for block in net.blocks_memory_retriever]
#     akl.put(grad_blocks_retriever=grad_mag)
#     grad_mag = [block.attn.out_proj.weight.grad.norm().item() for block in net.blocks_memory_creator]
#     akl.put(grad_blocks_creator=grad_mag)
#     if akl.ts%20==0:
#         for name, module in net.named_modules():
#             if isinstance(module, longrange.Block):
#                 key = f'attn_mat_{name}'
#                 akl.put(**{key: module.attn_weights.detach().cpu().numpy()})
    
# def print_network_gradient_stats(net):
#     print('Main blocks gradient mag:')
#     for block in net.blocks_main:
#         gradmag = block.attn.out_proj.weight.grad.norm().item()
#         print(f'{gradmag: .3e}', end=' ')
#     print()
#     print('Memory Retrieval blocks gradient mag:')
#     for block in net.blocks_memory_retriever:
#         gradmag = block.attn.out_proj.weight.grad.norm().item()
#         print(f'{gradmag: .3e}', end=' ')
#     print()
#     print('Memory Creator blocks gradient mag:')
#     for block in net.blocks_memory_creator:
#         gradmag = block.attn.out_proj.weight.grad.norm().item()
#         print(f'{gradmag: .3e}', end=' ')
#     print()
    
def main(args):
    print(f'Starting training with args: {args}')
    print('Loading dataset...')
    ds_train, ds_test = dataset.load_dataset(tqdm=tqdm)
    print('Done!')
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    config_net = longrange.get_config(args.model, max_context_length=args.seq_len)
    net = longrange.LongRangeGPT(**config_net)
    print(f'Model: {args.model}')
    print(f'Model config: {config_net}')
    print(f'Model # of parameters: {util.count_params(net)}')
    
    config = vars(args)
    config['config_net'] = config_net
    
    if args.track:
        run = wandb.init(config=config, name=args.name, save_code=True)

    n_seqs_train = 2 if net.config['use_memory'] else 1
    n_seqs_test = 4 if net.config['use_memory'] else 1
    train_longrange(net=net, ds=ds_train, n_batches=args.n_batches_train, batch_size=args.batch_size,
                    n_seqs=n_seqs_train, seq_len=args.seq_len, min_dist=args.min_dist, max_dist=args.max_dist,
                    lr=args.lr, device=args.device, wandb=wandb if args.track else None, tqdm=tqdm)
    loss_mean, loss_count = evaluate.evaluate_longrange(net=net, ds=ds_test, n_batches=args.n_batches_test, batch_size=args.batch_size,
                                                        n_seqs=n_seqs_test, seq_len=args.seq_len, min_dist=0, max_dist=1,
                                                        device=args.device, wandb=wandb if args.track else None, tqdm=tqdm)

    if args.track:
        for i_seq in range(len(loss_mean)):
            fig = evaluate.viz_losses(loss_mean[i_seq], loss_count[i_seq])
            plt.suptitle(f'Losses for sequence {i_seq}')
            wandb.log({f'viz loss': fig})

        net.eval().cpu()

        # make the log directory recursively
        os.makedirs(f'../results/{run.name}', exist_ok=True)
        torch.save(net, f'../results/{run.name}/model.pt')
        torch.save((loss_mean, loss_count), f'../results/{run.name}/losses.pt')

        run.finish()
        plt.close('all')

parser = argparse.ArgumentParser(description='Train a model.')

parser.add_argument('--model', type=str, default='longrange')
parser.add_argument('--n_batches_train', type=int, default=500)
parser.add_argument('--n_batches_test', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--seq_len', type=int, default=8)
parser.add_argument('--min_dist', type=int, default=0)
parser.add_argument('--max_dist', type=int, default=1)
parser.add_argument('--n_latent_tokens', type=int, default=None)

parser.add_argument('--lr', type=float, default=1e-3)

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--track', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--name', type=str, default=None)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

"""
# Test

python train.py --model transformer --n_batches_train 0100 --n_batches_test 0100 --batch_size 64 --seq_len 8 --seed 0 --lr 1e-3 --device cuda:0 --track --name testing

# Learning Rate Sweep

# Transformer Model Size Sweep

# Longrange Model Size Sweep

# Seq Length Sweep

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

