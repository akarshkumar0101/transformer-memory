
import torch
from torch import nn

import numpy as np

class HopfieldMemory(nn.Module):
    def __init__(self, shape, alpha=0., opt='sgd', lr=1.):
        """
        shape is (..., n_patterns, embed_dim)
        """
        super().__init__()
        
        # keys/stored patterns
        self.K = nn.Parameter(torch.randn(*shape))
        # values
        self.V = nn.Parameter(torch.randn(*shape))
        
        self.alpha = alpha
        if opt=='sgd':
            self.opt = torch.optim.SGD(self.parameters(), lr=lr, maximize=True)
        elif opt=='adam':
            self.opt = torch.optim.Adam(self.parameters(), lr=lr, maximize=True)
        
    def set_target(self, K_target, V_target=None):
        self.opt.zero_grad()
        self.K.grad = (1.-self.alpha)*(K_target-self.K)
        self.V.grad = (1.-self.alpha)*(V_target-self.V) if V_target is not None else None
    
    def step(self):
        self.opt.step()
        
    def set_target_with_data(self, xk, xv=None, beta=1., dist_metric='dot', beta2=None):
        """
        K.shape is (..., k, d)
        xk.shape is (..., n, d) is the data's queries
        xv.shape is (..., n, d) is the data's values
        """
        if dist_metric=='dot':
            dists = self.K@xk.transpose(-1, -2)
        elif dist_metric=='euclidean':
            dists = -(self.K[..., None, :] - xk).norm(dim=-1)
            
        # dists.shape is (..., k, n)
        
        dk = (beta*dists).softmax(dim=-2)
        # dx = (beta*dists).softmax(dim=-1)
        if beta2 is None:
            dkx = dk/dk.sum(dim=-1, keepdim=True)
        else:
            dkx = (beta2*dk).softmax(dim=-1)
        
        K_target = dkx@xk
        V_target = dkx@xv if xv is not None else None
        
        self.set_target(K_target, V_target)

    def set_target_attn(self, attn_weights, query, key, value, beta=1., dist_metric='dot', beta2=None):
        """
        K.shape is (..., k, d)
        xk.shape is (..., n, d) is the data's queries
        xv.shape is (..., n, d) is the data's values

        attn_weights is     (..., n, k+n) before softmax
        query is            (..., n, d)
        key   is            (..., k+n, d)
        value is            (..., k+n, d)
        """
        xk = query.detach()
        xv = None

        klen = self.K.shape[-2]
        nlen = query.shape[-2]
        dlen = self.K.shape[-1]

        A = attn_weights
        A_mem, A_context = attn_weights.split([klen, nlen], dim=-1)
        A_sm = A.softmax(dim=-1)
        A_mem_sm, A_context_sm = A_mem.softmax(dim=-1), A_context.softmax(dim=-1)

        Q, K, V = query, key, value
        K_mem, K_context = key.split([klen, nlen], dim=-2)
        V_mem, V_context = value.split([klen, nlen], dim=-2)

        # sanity checks
        # assert (self.K-key[..., :klen, :]).abs().max()<1e-4
        # assert (A_mem-(query@self.K.transpose(-1, -2)/np.sqrt(dlen))).abs().max()<1e-4

        # A_mem is of shape (..., n, k)

        if dist_metric=='dot':
            # dists = self.K@xk.transpose(-1, -2)
            dists = A_mem
        elif dist_metric=='euclidean':
            dists = -(self.K[..., None, :] - xk).norm(dim=-1)
            
        dists = dists.transpose(-1, -2)
        # dists.shape is (..., k, n)
        
        dk = (beta*dists).softmax(dim=-2)
        # dx = (beta*dists).softmax(dim=-1)
        if beta2 is None:
            dkx = dk/dk.sum(dim=-1, keepdim=True)
        else:
            dkx = (beta2*dk).softmax(dim=-1)
        
        K_target = dkx@xk
        V_target = dkx@xv if xv is not None else None
        
        self.set_target(K_target, V_target)