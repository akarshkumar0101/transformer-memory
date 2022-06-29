from tkinter import N
from matplotlib import use
from matplotlib.style import context
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
import wandb

from hopfield_memory import HopfieldMemory
import viz
from utils import *

from transformers import AutoTokenizer
from modeling_gpt2 import GPT2Model, GPT2LMHeadModel

from tqdm import tqdm

with open('murder.txt') as f:
    text_book = f.read()
    text_book = ' '.join(text_book.split())

# def experiment(tokenizer, model, text=None, context_length=100, stride=1, device='cpu', wandb=None, do_viz=True, tqdm=None, **kwargs):
def experiment(args):
    device = torch.device(args.device if args.device is not None else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(device)
    torch.manual_seed(args.seed)
        
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    model = model.to(device)

    text = text_book
    tokens_all = tokenizer.tokenize(text)
    input_ids_all = tokenizer(text, return_tensors='pt').input_ids[0]
    
    n_tokens = args.n_tokens if args.n_tokens is not None else len(input_ids_all)
    n_context = args.n_context
    n_memories = args.n_memories
    idx_layer_viz = args.idx_layer_viz
    idx_head_viz = args.idx_head_viz
    n_viz_steps = args.n_viz_steps
    stride = args.stride
    memory_type = args.memory_type
    use_pos = args.use_pos
    
    steps = defaultdict(lambda: [])
    nlls = []

    idx_token_start = n_context+n_memories
    idx_token_end = min(len(input_ids_all), idx_token_start+n_tokens)
    n_tokens = idx_token_end-idx_token_start
    pbar = range(idx_token_start, idx_token_end, stride)

    if tqdm is not None:
        pbar = tqdm(pbar)

    for idx_loop, idx_token in enumerate(pbar):
        step = {'idx_loop': idx_loop}

        begin_loc = max(idx_token + stride - n_context, 0)
        end_loc = min(idx_token + stride, len(input_ids_all))
        trg_len = end_loc - idx_token  # may be different from stride on last loop
        # input_ids = input_ids_all[None, begin_loc: end_loc].to(device)
        # input_ids = input_ids_all[None, :context_length].to(device) # REMOVE THIS LINE
        # tokens_context = tokens_all[begin_loc: end_loc]
        # target_ids = input_ids.clone()
        # target_ids[:, :-trg_len] = -100

        input_ids_context = input_ids_all[None, begin_loc: end_loc].to(device)
        input_ids_memory = input_ids_all[None, begin_loc-n_memories: begin_loc].to(device)

        # step['tokens_context'] = ''.join(tokens_all[begin_loc: end_loc]).replace('Ġ', ' ')

        target_ids = input_ids_context.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():

            ak = {'debug': True, 'use_pos': use_pos}
            if memory_type is None or memory_type=='None':
                ak['pKpV'] = None
            else:
                akp = {'debug': True, 'use_pos': use_pos}
                _ = model(input_ids_memory, ak=akp)
                Q, K, V = akp['QKV']
                O = akp['O']
                A = Q@K.transpose(-1, -2)
                A_sm = sm(A, 1/8., -1)

                if memory_type=='KV':
                    ak['pKpV'] = K.clone(), V.clone()
                elif memory_type=='QV':
                    ak['pKpV'] = Q.clone(), V.clone()
                elif memory_type=='AQAV':
                    ak['pKpV'] = Q.clone(), V.clone()


            outputs = model(input_ids_context, labels=target_ids, ak=ak)
            nll_i = outputs[0] * trg_len
            step['nll_i'] = nll_i.item()
            nlls.append(nll_i.item())
            step['nll_avg'] = np.mean(nlls)
            step['ppl_avg'] = np.e**np.mean(nlls)

        if args.wandb:
            wandb.log(step)

        # if n_viz_steps is not None and idx_loop%(n_tokens//n_viz_steps)==0:
            # viz.viz_Am(Am[idx_layer_viz, 0, idx_head_viz], beta1=beta1, beta2=beta2)
            # plt.suptitle(f'layer {idx_layer_viz} head {idx_head_viz}'); plt.tight_layout(); plt.show()
            # viz.viz_Q_Km(Q[idx_layer_viz, 0, idx_head_viz], Km[idx_layer_viz, 0, idx_head_viz])
            # plt.suptitle(f'layer {idx_layer_viz} head {idx_head_viz}'); plt.tight_layout(); plt.show()

            # Q_unique, K_unique, Km_unique = calc_n_unique_vectors(Q[:, 0]), calc_n_unique_vectors(K[:, 0]), calc_n_unique_vectors(Km[:, 0])
            # plt.figure(figsize=(20, 5))
            # plt.subplot(131); viz.imshow_unique_vecs(Q_unique)
            # plt.subplot(132); viz.imshow_unique_vecs(K_unique)
            # plt.subplot(133); viz.imshow_unique_vecs(Km_unique)
            # plt.suptitle('Q, K, Km'); plt.show()


            # plt.figure(figsize=(20, 10))
            # A = Q@K.transpose(-1, -2)
            # Am = Q@Km.transpose(-1, -2)
            # a = torch.cat([Am, A], dim=-1)[idx_layer_viz, 0, idx_head_viz]
            # plt.imshow(to_np(a)); plt.colorbar()
            # plt.show()


        # do_step_viz(**locals())
        # a = (Km[..., None, :, :]-Km[..., :, None, :]).norm(dim=-1)
        # step['Km_spread'] = a.mean().item()
        # a = (Q[..., None, :, :]-Q[..., :, None, :]).norm(dim=-1)
        # step['Q_spread'] = a.mean().item()

        # step['Am_alpha'] = softmax(Am[idx_layer_viz, 0, idx_head_viz], beta1, -1).mean(dim=-2)

        # for key, value in step.items():
            # steps[key].append(value)
            
    # plt.figure(figsize=(20, 5))
    # a = np.array([steps[f'dK_layer{layer_idx}'] for layer_idx in range(6)]).T
    # plt.plot(a, label=[f'dK_layer{layer_idx}' for layer_idx in range(6)])
    # plt.legend()
    # plt.show()
            
    # print(steps.keys())
    # plt.figure(figsize=(20,5))
    # plt.plot(steps['Km_spread'], label='Km spread')
    # plt.plot(steps['Q_spread'], label='Q spread')
    # plt.legend()
    # plt.show()

    # a = torch.stack(steps['Am_alpha'])
    # plt.figure(figsize=(20, 5))
    # plt.plot(to_np(a[:, :]))
    # plt.title(f'memory alphas over time for Layer {idx_layer_viz} head {idx_head_viz}')
    # plt.ylabel('Computed alpha')
    # plt.xlabel('Time step')

    

import argparse
parser = argparse.ArgumentParser(description='Experiment to test the effects of different key/value memories')

parser.add_argument('--n_context', type=int)
parser.add_argument('--n_memories', type=int)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--memory_type', type=str)
parser.add_argument('--n_tokens', type=int, default=None)
parser.add_argument('--wandb', default=False, action='store_true')
parser.add_argument('--n_viz_steps', type=int, default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=str, default=None)
parser.add_argument('--use_pos', default=False, action='store_true')

parser.add_argument('--idx_layer_viz', type=int, default=3)
parser.add_argument('--idx_head_viz', type=int, default=0)

def main():
    args = parser.parse_args()
    print(args)
    if args.wandb:
        wandb.init()
        wandb.config.update(args)

    experiment(args)

    if args.wandb:
        wandb.finish()

if __name__=='__main__':
    main()