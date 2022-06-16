import torch
import numpy as np

def to_np(a):
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    elif isinstance(a, np.ndarray):
        return a
    else:
        return np.array(a)
    
def count_params(net):
    return np.sum([p.numel() for p in net.parameters()], dtype=int)

"""
alpha= 0.0 will take the mean.
alpha= 0.2 will be very close to max.
alpha=-0.2 will be very close to min.
"""
def smooth_max(x, alpha, dim=-1):
    # unstable version:
    # return (x*(alpha*x).exp()).sum()/((alpha*x).exp()).sum()
    return ((alpha*x).softmax(dim=dim)*x).sum(dim=dim)


def softmax(x, beta=1., dim=-1):
    if beta=='weighted':
        return x/x.sum(dim=dim, keepdim=True)
    else:
        return (beta*x).softmax(dim=dim)
    