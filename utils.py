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

