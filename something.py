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

with open('book.txt') as f:
    text_book = f.read()
    text_book = ' '.join(text_book.split())


def initialize_hms(**kwargs):
    n_memories, alpha, use_adaptive_alpha, beta1, beta2, beta3, device = \
        kwargs['n_memories'], kwargs['alpha'], kwargs['use_adaptive_alpha'], \
        kwargs['beta1'], kwargs['beta2'], kwargs['beta3'], kwargs['device']
    input_ids_all = kwargs['input_ids_all']
    context_length = kwargs['context_length']
    model = kwargs['model']
    idx_layer_viz = kwargs['idx_layer_viz']

    print('Initializing Hopfield memories...')
    print('alpha:', alpha)
    print('use_adaptive_alpha:', use_adaptive_alpha)

    n_layers, n_heads = model.config.n_layer, model.config.n_head
    n_dim = model.config.n_embd//n_heads

    # hms = HopfieldMemory((n_layers, 1, n_heads, n_memories, n_dim), alpha=alpha, use_adaptive_alpha=use_adaptive_alpha, opt='sgd', lr=1.0).to(device)
    hms = [HopfieldMemory((1, 12, n_memories, 64), alpha=alpha, use_adaptive_alpha=use_adaptive_alpha, opt='sgd', lr=1.0).to(device) for i in range(6)]

    idx_token_start = 1000
    input_ids = input_ids_all[None, idx_token_start:idx_token_start+context_length].to(device)
    ak = {'debug': True, 'layers': [], 'hms': hms}
    outputs = model(input_ids, labels=None, ak=ak)

    Q, K, V = ak['layers'][idx_layer_viz]['QKV']
    hm = hms[idx_layer_viz]
    A = Q@K.transpose(-1, -2)
    Am = Q@hm.Km.transpose(-1, -2)

    plt.figure(figsize=(20, 5))
    plt.subplot(121); plt.title('Km~N(0,1), K~LM')
    plt.hist([to_np(A.flatten()), to_np(Am.flatten())], bins=100, label=['A', 'Am']); plt.legend()


    # Q, K, V = (torch.stack([akl['QKV'][i] for akl in ak['layers']]) for i in range(3))
    # hm.reset(1.)

    for idx_layer, hm in enumerate(hms):
        Q, K, V = ak['layers'][idx_layer]['QKV']
        hm.to(Q)
        # hm.reset(1.)
        # hm.Km.data[...] = hm.Km.data[...]*K.std(dim=-2, keepdim=True)+K.mean(dim=-2, keepdim=True)
        # hm.Km.data[:, :] = (hm.Km.data[:, :]+K.mean(dim=0))*K.std(dim=0)
        # hm.Km.data[:, :] = Q[torch.randint(len(Q), (len(hm.Km), ))]

    Q, K, V = ak['layers'][idx_layer_viz]['QKV']
    hm = hms[idx_layer_viz]
    A = Q@K.transpose(-1, -2)
    Am = Q@hm.Km.transpose(-1, -2)
    print(A.shape, Am.shape)
    plt.subplot(122); plt.title('Km~N(K.mean, K.std), K~LM')
    plt.hist([to_np(A.flatten()), to_np(Am.flatten())], bins=100, label=['A', 'Am']); plt.legend()
    plt.suptitle('Distribution of values in matrices')
    plt.show()

    return hms

def experiment(tokenizer, model, text=None, context_length=100, stride=1, device='cpu', wandb=None, do_viz=True, tqdm=None, **kwargs):
    if text is None:
        text = text_book
        
    torch.manual_seed(11)
    tokens_all = tokenizer.tokenize(text)
    input_ids_all = tokenizer(text, return_tensors='pt').input_ids[0]
    
    model = model.to(device)

    beta1, beta2 = 1/8., 100.
    # beta1, beta2 = 1/50., 100.
    beta1, beta2 = 1/50., 1000.
    beta3 = None
    # alpha = 0.1
    # use_adaptive_alpha = False
    alpha = 0.1
    use_adaptive_alpha = True

    n_memories = 80
    idx_head_viz = 0
    idx_layer_viz = 3
    n_viz_steps = 4 # number of steps to visualize
    
    hms = initialize_hms(**locals())
        
    steps = defaultdict(lambda: [])

    idx_token_start = context_length+1000
    idx_token_end = len(input_ids_all)
    idx_token_end = idx_token_start+10000 # REMOVE THIS LINE
    len_loop = idx_token_end-idx_token_start
    pbar = range(idx_token_start, idx_token_end, stride)
    if tqdm is not None:
        pbar = tqdm(pbar)
    for idx_loop, idx_token in enumerate(pbar):
        step = {'idx_loop': idx_loop}

        begin_loc = max(idx_token + stride - context_length, 0)
        end_loc = min(idx_token + stride, len(input_ids_all))
        trg_len = end_loc - idx_token  # may be different from stride on last loop
        input_ids = input_ids_all[None, begin_loc: end_loc].to(device)
        input_ids = input_ids_all[None, :context_length].to(device) # REMOVE THIS LINE
        tokens_context = tokens_all[begin_loc: end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        step['tokens_context'] = tokens_context
        
        with torch.no_grad():
            ak = {'debug': True, 'layers': [], 'hms': hms}
            outputs = model(input_ids, labels=target_ids, ak=ak)
            nll_i = outputs[0] * trg_len
            step['nll_i'] = nll_i.item()
            
        for idx_layer, hm in enumerate(hms):
            akl = ak['layers'][idx_layer]
            A, A_sm, O, (Q, K, V) = akl['A'], akl['A_sm'], akl['O'], akl['QKV']
            
            Am = Q@hm.Km.transpose(-1, -2)
            QQ = Q@Q.transpose(-1, -2)
            QK = Q@K.transpose(-1, -2)
            # print(Am.min().item(), Am.mean().item(), Am.max().item())
            
            Km_target = Q
            # Km_target = A_sm@K
            hm.set_target_with_data(Q=Km_target, O=O, dist_metric='dot', 
                                    beta1=beta1, beta2=beta2 , beta3=None)
            hm.step()

        Q, K, V = ak['layers'][idx_layer_viz]['QKV']
        Km = hms[idx_layer_viz].Km
        if do_viz and idx_loop%(len_loop//n_viz_steps)==0:
            Am = Q@Km.transpose(-1, -2)
            print(Am.shape, idx_head_viz)
            viz.viz_Am(Am[0, idx_head_viz], beta1=beta1, beta2=beta2)
            plt.suptitle(f'layer {idx_layer_viz} head {idx_head_viz}'); plt.tight_layout(); plt.show()
            viz.viz_Q_Km(Q[0, idx_head_viz], Km[0, idx_head_viz])
            plt.suptitle(f'layer {idx_layer_viz} head {idx_head_viz}'); plt.tight_layout(); plt.show()
        # do_step_viz(**locals())
        a = (Km[..., None, :, :]-Km[..., :, None, :]).norm(dim=-1)
        step['Km_spread'] = a.mean().item()
        a = (Q[..., None, :, :]-Q[..., :, None, :]).norm(dim=-1)
        step['Q_spread'] = a.mean().item()

        for key, value in step.items():
            steps[key].append(value)
            
    # plt.figure(figsize=(20, 5))
    # a = np.array([steps[f'dK_layer{layer_idx}'] for layer_idx in range(6)]).T
    # plt.plot(a, label=[f'dK_layer{layer_idx}' for layer_idx in range(6)])
    # plt.legend()
    # plt.show()
            
    print(steps.keys())
    plt.figure(figsize=(20,5))
    plt.plot(steps['Km_spread'], label='Km spread')
    plt.plot(steps['Q_spread'], label='Q spread')
    plt.legend()
    plt.show()
    return hms
    
#     plt.figure(figsize=(20,5))
#     graph = np.array(graph).swapaxes(-1, -2)
#     plt.plot(graph[:, 0, 1], c='r', label='Am values')
#     plt.plot(graph[:, 1, 1], c='g', label='QQ values')
#     plt.plot(graph[:, 2, 1], c='b', label='Qk values')
#     # plt.plot(np.array(graph), label=['Am', 'QQ', 'QK'])
#     plt.legend()
#     plt.show()

# experiment(tokenizer, model, context_length=20, stride=20, device=device, do_viz=True)

def do_step_viz(**kwargs):
    step = kwargs['step']
    layer_idx_viz = kwargs['idx_layer_viz']
    hm = kwargs['hms'][layer_idx_viz]
    ua = hm.used_alpha
    step[f'ua_layer{layer_idx_viz}'] = ua

    idx_loop = kwargs['idx_loop']
    len_loop = kwargs['idx_token_end'] - kwargs['idx_token_start']
    n_viz_steps = kwargs['n_viz_steps']
    do_viz = kwargs['do_viz'] and idx_loop%(len_loop//n_viz_steps)==0
    print(do_viz)

    if do_viz:
        viz.viz_Am(Am[0,0], beta1=beta1, beta2=beta2)
        plt.tight_layout(); plt.show()
        viz.viz_Q_Km(Q[0,0], Km[0,0])
        plt.tight_layout(); plt.show()
        
    Kmp = Km.clone()
    # dK = (Km-Kmp).norm(dim=-1).mean().item()
    # steps[f'dK_layer{layer_idx}'].append(dK)
        
def calc_n_unique_vectors(X, distance=2.5, method='smart', do_viz_hist=False, do_viz_mat=False):
    """
    X.shape should be (..., n_vectors, n_dim)
    """
    a = (X[..., :, None, :]-X[..., None, :, :]).norm(dim=-1)
    
    if do_viz_hist:
        plt.hist((to_np(a.flatten())), bins=100)
        plt.gca().axvline(distance, c='r', linestyle='dotted')
        plt.show()

    a = (a<distance)

    if do_viz_mat:
        plt.imshow(to_np(a.to(float)), vmin=0, vmax=1); plt.colorbar()
        plt.show()
        
    if method=='force':
        raise Exception("does not actually work with batches so don't use")
        n = 0
        while len(a)>0:
            n += 1 # found a unique vector
            idxs = torch.ones(len(a)).to(a).to(bool)
            idxs[a[0]] = False # remove all vectors that match me, including me
            a = a[idxs][:, idxs]
    
    elif method=='smart':
        n = (1./a.sum(dim=-1)).sum(dim=-1)
    
    return n

