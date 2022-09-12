import torch
from utils import to_np, softmax as sm

import matplotlib.pyplot as plt

import numpy as np

from matplotlib.colors import LogNorm


def plt_grid(mat_shape):
    # c = [.5, .5, .5, .2]
    c = [1., 0., 0., .1]
    for i in range(mat_shape[0]):
        plt.axhline(i, c=c)
    for i in range(mat_shape[1]):
        plt.axvline(i, c=c)

def viz_Am(Am, beta1, beta2, figsize=(20, 5), grid=True, vmax=None, log_dist=False):
    cl, m = Am.shape
    a = sm(Am, beta1, -1)
    b = sm(sm(Am, beta1, -1), beta2, -2)
    c = sm(Am, beta1, -1).mean(dim=-2)
    def plot_adaptive_step_size_over():
        # ax = plt.gca()
        # ax.set_ylim(Am.shape[0]+.5, -10.)
        # ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        # ax2.set_ylabel('Adaptive step size', color='r')  # we already handled the x-label with ax1
        # ax2.plot(to_np(c), color='r')
        # ax2.tick_params(axis='y', labelcolor='r')
        # ax2.set_ylim(0, 1)
        plt.plot(to_np(c)*-cl+cl, color='r')
        # plt.text(100, 10, 'Adaptive step size', rotation=90, color='r')
        
    plt.figure(figsize=figsize)
    plt_size = (1, 3)
    # , shading='auto', 
    
    plt.subplot(*plt_size, 1); plt.title('Am')
    plt.imshow(to_np(Am)); plt.colorbar()
    if grid:
        plt_grid(Am.shape)
    plt_args = dict(vmin=None, vmax=vmax)
    
    plt.subplot(*plt_size, 2); plt.title(f'Am.sm({beta1}, -1)')
    plt.imshow(to_np(a), norm=LogNorm() if log_dist else None, **plt_args); plt.colorbar()
    plot_adaptive_step_size_over()
    if grid:
        plt_grid(Am.shape)
    
    plt.subplot(*plt_size, 3); plt.title(f'Am.sm({beta1}, -1).sm({beta2}, -2)')
    plt.imshow(to_np(b), norm=LogNorm() if log_dist else None, **plt_args); plt.colorbar()
    plot_adaptive_step_size_over()
    if grid:
        plt_grid(Am.shape)
    
    # plt.imshow(to_np(a), vmin=0, vmax=1); plt.colorbar()
    # y, x = np.arange(Am.shape[0])[::-1], np.arange(Am.shape[1])
    # y, x = np.meshgrid(y, x, indexing='ij')
    # plt.pcolormesh(to_np(a), x, y, vmin=0, vmax=1); plt.colorbar()

def viz_Q_Km(Q, Km, figsize=(10, 5), grid=True):
    plt.figure(figsize=figsize)
    plt.subplot(121); plt.title('Q')
    plt.imshow(to_np(Q))
    if grid:
        plt_grid(Q.shape)
    plt.subplot(122); plt.title('Km')
    plt.imshow(to_np(Km))
    if grid:
        plt_grid(Km.shape)

def imshow_unique_vecs(a):
    """
    a.shape should be (layers, heads)
    """
    plt.title("Number of unique vectors")
    plt.ylabel('# of Layers'); plt.xlabel('# of Heads')
    plt.imshow(to_np(a)); plt.colorbar()
    
    
def subplot_layers_heads(fn_plot, n_layers=6, n_heads=12, figsize=(20, 10),
                         constrained_layout=True, sharex=True, sharey=True):
    fig, axs = plt.subplots(n_layers, n_heads, figsize=figsize,
                            constrained_layout=constrained_layout, sharex=sharex, sharey=sharey)
    axs = np.array(axs)
    while axs.ndim<2:
        axs = axs[..., None]

    for idx_layer in range(n_layers):
        for idx_head in range(n_heads):
            ax = axs[idx_layer, idx_head]
            plt.sca(ax)
            fn_plot(idx_layer, idx_head)
    fig.supxlabel('Heads', fontsize=30)
    fig.supylabel('Layers', fontsize=30)

    # plt.tight_layout()
    return fig

def viz_A_Am_distribution(A, Am):
    subplot_layers_heads(
        lambda l, h: [plt.hist([to_np(A[l, h].flatten()), to_np(Am[l, h].flatten())], bins=30, label=['A', 'Am'])],
        # 1,1,
        sharex=False, sharey=False,
    )
    plt.legend()
    plt.suptitle('Distribution of values of A=Q@K.T and Am=Q@Km.T', fontsize=30)