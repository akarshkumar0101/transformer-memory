import torch
import numpy as np
import matplotlib.pyplot as plt

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

sm = softmax


def calc_n_unique_vectors(X, distance=2.5, method='smart', do_viz_hist=False, do_viz_mat=False):
    """
    X.shape should be (..., n_vectors, n_dim)
    """
    mat = (X[..., :, None, :]-X[..., None, :, :]).norm(dim=-1)
    
    if do_viz_hist:
        plt.hist((to_np(mat.flatten())), bins=100)
        plt.gca().axvline(distance, c='r', linestyle='dotted')
        plt.show()

    mat_bin = (mat<distance)

    if do_viz_mat:
        plt.figure(figsize=(20, 10))
        plt.subplot(121)
        plt.imshow(to_np(mat.to(float))); plt.colorbar()
        plt.subplot(122)
        plt.imshow(to_np(mat_bin.to(float)), vmin=0, vmax=1); plt.colorbar()
        plt.show()
        
    if method=='force':
        raise Exception("does not actually work with batches so don't use")
        n = 0
        while len(a)>0:
            n += 1 # found a unique vector
            idxs = torch.ones(len(mat_bin)).to(mat_bin).to(bool)
            idxs[mat_bin[0]] = False # remove all vectors that match me, including me
            mat_bin = mat_bin[idxs][:, idxs]
    
    elif method=='smart':
        n = (1./mat_bin.sum(dim=-1)).sum(dim=-1)
    
    return n