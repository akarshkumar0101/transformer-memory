import torch
from utils import to_np, softmax as sm

import matplotlib.pyplot as plt

import numpy as np


def viz_Am(Am, beta1, beta2, figsize=(20, 5)):
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
    
    plt.subplot(*plt_size, 1); plt.title('Am')
    plt.imshow(to_np(Am)); plt.colorbar()
    
    plt.subplot(*plt_size, 2); plt.title(f'Am.sm({beta1}, -1)')
    plt.imshow(to_np(a), vmin=0, vmax=1); plt.colorbar()
    plot_adaptive_step_size_over()
    
    plt.subplot(*plt_size, 3); plt.title(f'Am.sm({beta1}, -1).sm({beta2}, -2)')
    plt.imshow(to_np(b), vmin=0, vmax=1); plt.colorbar()
    plot_adaptive_step_size_over()
    
    # plt.imshow(to_np(a), vmin=0, vmax=1); plt.colorbar()
    # y, x = np.arange(Am.shape[0])[::-1], np.arange(Am.shape[1])
    # y, x = np.meshgrid(y, x, indexing='ij')
    # plt.pcolormesh(to_np(a), x, y, vmin=0, vmax=1); plt.colorbar()

def viz_Q_Km(Q, Km, figsize=(10, 5)):
    plt.figure(figsize=figsize)
    plt.subplot(121); plt.title('Q')
    plt.imshow(to_np(Q))
    plt.subplot(122); plt.title('Km')
    plt.imshow(to_np(Km))
    plt.show()