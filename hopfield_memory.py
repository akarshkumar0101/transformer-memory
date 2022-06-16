
import torch
from torch import nn

import numpy as np

from utils import smooth_max, softmax as sm

class HopfieldMemory(nn.Module):
    def __init__(self, shape, alpha=0.5, use_adaptive_alpha=False, opt='sgd', lr=1.):
        """
        shape is (..., m, d)
        d is the embedding dimension
        m is the number of memory keys and values

        if use_adaptive_alpha is False:
            alpha = 0 means no movement
            alpha = 1 means teleport to target
        if use_adaptive_alpha is True:
            alpha serves as a scale for the adaptive alpha
        
        opt is the optimizer
        lr is the learning rate for the optimizer
        """
        super().__init__()

        # keys/stored patterns
        self.Km = nn.Parameter(torch.randn(*shape)*.1)
        self.Km.requires_grad_(False)
        # values
        self.Vm = nn.Parameter(torch.randn(*shape)*.1)
        self.Vm.requires_grad_(False)
        
        self.alpha = alpha
        self.use_adaptive_alpha = use_adaptive_alpha
        if opt=='sgd':
            self.opt = torch.optim.SGD(self.parameters(), lr=lr, maximize=True)
        elif opt=='adam':
            self.opt = torch.optim.Adam(self.parameters(), lr=lr, maximize=True)

    def reset(self, mag=0.1):
        self.Km.data[...] = mag*torch.randn_like(self.Km)
        self.Vm.data[...] = mag*torch.randn_like(self.Vm)
        
    # @torch.no_grad()
    def set_target(self, Km_target, Vm_target=None, alpha=None):
        if alpha is None:
            alpha = self.alpha
        if isinstance(alpha, torch.Tensor) and alpha.shape==(len(self.Km), ):
            alpha = alpha[:, None]

        self.opt.zero_grad()

        self.Km.grad = alpha*(Km_target-self.Km)
        self.Vm.grad = alpha*(Vm_target-self.Vm) if Vm_target is not None else None
    
    def step(self):
        self.opt.step()
        
    # @torch.no_grad()
    def set_target_attn(self, Am, A, Q, K, V, O, 
                        beta1=1., beta2=1., beta3=1.):
        """
        Am is (..., c, m)
        A is (..., c, c)

        Am is the similarity matrix between the queries and memory keys
        A is the similarity matrix between the queries and context keys
        Am, A are before softmax applied

        Q, K, V, O are (..., c, d)
        K, V are unused
        self.Km's target will be in the convex hull of Q
        self.Vm's target will be in the convex hull of O

        self.Km is (..., m, d)
        self.Vm is (..., m, d)
        """
        # c, d = Q.shape[-2:]
        # m = self.Km.shape[-2]

        # Am_sm, A_sm = Am.softmax(dim=-1), A.softmax(dim=-1)
        # A_full_sm = torch.cat([Am, A], dim=-1).softmax(dim=-1) # (..., c, m+c)

        # import wandb
        # wandb.log({"Am_min": Am.min().item(), "Am_max": Am.max().item(), "Am_mean": Am.mean().item()})

        Dc = sm(Am, beta1, -1) # (..., c, m)
        Dc = sm(Dc, beta2, -2)
        Dc = Dc.transpose(-1, -2)
        
        Km_target = Dc@Q # (..., m, d)
        Vm_target = Dc@O if O is not None else None # (..., m, d)

        alpha = self.alpha
        if self.use_adaptive_alpha:
            alpha = alpha*smooth_max(sm(Am, beta1, -1), alpha=0., dim=-2) # (..., m)
            # alpha = sm(alpha, beta3, -1)

        self.set_target(Km_target, Vm_target, alpha=alpha)
        self.Am = Am
        
    # @torch.no_grad()
    def set_target_with_data(self, Q, O=None, dist_metric='dot',
                             beta1=1., beta2=1., beta3=1.):
        """
        Am is (..., c, m)
        self.Km is (..., m, d)
        self.Vm is (..., m, d)

        Q, K, V, O are (..., c, d)
        K, V are unused
        self.Km's target will be in the convex hull of Q
        self.Vm's target will be in the convex hull of O

        self.Km is (..., m, d)
        self.Vm is (..., m, d)
        """

        if dist_metric=='dot':
            Am = Q@self.Km.transpose(-1, -2)
        elif dist_metric=='euclidean':
            Am = -torch.linalg.norm(Q[..., :, None, :] - self.Km[..., None, :, :], dim=-1)
        
        self.set_target_attn(Am, None, Q, None, None, O, beta1, beta2, beta3)
        