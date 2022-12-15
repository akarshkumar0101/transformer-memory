import dataset
import longrange
import matplotlib.pyplot as plt
import torch
from matplotlib import cm, colors
from torch import nn
from tqdm.auto import tqdm

# def evaluate_transformer(ds, net, n_batches=10, batch_size=32, n_seqs=4, seq_len=100, device='cpu', wandb=None, tqdm=None):
#     loss_fn = nn.CrossEntropyLoss(reduction='none')
    
#     net.eval().to(device)
#     # average loss
#     # average loss across sequence
#     # average loss for (first character in word)
#     # average loss for (different freq bins)
    
#     with torch.no_grad():
#         data = {}
#         losses = [[] for _ in range(n_seqs)]

#         pbar = dataset.get_random_batches_new(ds, n_batches, batch_size, n_seqs, seq_len, min_dist=0, max_dist=1, unbind=True)
#         for batch_ids, batch_freqs in tqdm(pbar, total=n_batches):
#             memory = None
#             for idx_seq, (seq_ids, seq_freqs) in enumerate(zip(batch_ids, batch_freqs)):
#                 seq_ids, seq_freqs = seq_ids.to(device).long(), seq_freqs.to(device).long()
#                 logits_i, memory_i = net.forward(seq_ids, memory_in=memory, calc_memory_out=True)
#                 # print(seq_ids.shape, seq_freqs.shape)
#                 memory = memory_i if memory is None else torch.cat([memory, memory_i], dim=-2)
#                 # print(memory.shape)

#                 a = logits_i[..., :-1, :].reshape(-1, logits_i.shape[-1])
#                 b = seq_ids[..., 1:].reshape(-1)
#                 loss = loss_fn(a, b).reshape(batch_size, -1).mean(dim=0).detach().cpu()
#                 # print(loss.shape)
#                 losses[idx_seq].append(loss)
#         return losses


# def evaluate_longrange(ds, net, n_batches=10, batch_size=32, n_seqs=4, seq_len=100, device='cpu', wandb=None, tqdm=None):
#     loss_fn = nn.CrossEntropyLoss(reduction='none')
    
#     net.eval().to(device)
    
#     losses_all = []
#     fbin2loss_all = []
    
#     pbar = dataset.get_seq_batches(ds=ds, n_batches=n_batches, batch_size=batch_size, n_seqs=n_seqs,
#                                    seq_len=seq_len, min_dist=0, max_dist=1, unbind=False,
#                                    device=device, dtype=torch.long)
#     pbar = tqdm(pbar, total=n_batches) if tqdm else pbar
    
#     with torch.no_grad():
#         for batch_ids, batch_fbin in pbar:
#             losses, fbin2loss = longrange.loss_fn_longrange(net, batch_ids, batch_fbin)
#             print(losses.shape, fbin2loss.shape)
#             fbin2loss_all.append(fbin2loss)
#             losses_all.append(losses.mean(dim=0).detach().cpu())
#         losses_all = torch.stack(losses_all)
#     fbin2loss_all = {key: torch.stack([fbin2loss[key] for fbin2loss in fbin2loss_all]).mean(dim=0) for key in fbin2loss_all[0].keys()}
#     # losses is of shape (n_seq, seq_len-1)
#     # fbin2loss_all is dictionary from fbin to array of size n_seq
#     return losses_all.mean(dim=0), fbin2loss_all

def viz_losses(loss_mean, loss_count):
    """
    loss_mean: (seq_len, 10)
    loss_count: (seq_len, 10)
    """
    plt.figure(figsize=(10, 7.5))
    plt.subplot(321); plt.title('PPL mean'); plt.ylabel('fbin'); plt.xlabel('seq_len')
    plt.imshow(loss_mean.exp().detach().cpu().numpy().T)
    plt.colorbar()
    plt.subplot(322); plt.title('PPL count'); plt.ylabel('fbin'); plt.xlabel('seq_len')
    plt.imshow(loss_count.detach().cpu().numpy().T, norm=colors.LogNorm())
    plt.colorbar()

    plt.subplot(312)
    plt.title('PPL vs seq_len for different fbin')
    cmap = plt.get_cmap('autumn') 
    cNorm  = colors.Normalize(vmin=0, vmax=7)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)
    plt.plot(loss_mean.exp().detach().cpu().numpy()[:, 0], label='Within word', color='b')
    for i in range(8):
        plt.plot(loss_mean.exp().detach().cpu().numpy()[:, i+2], label=f'rarity: {i}', color=scalarMap.to_rgba(i))
    ylim = plt.gca().get_ylim()
    plt.plot(loss_mean.exp().detach().cpu().numpy()[:, 1], label='unknown word', color='k')
    plt.gca().set_ylim(ylim)
    plt.legend()

    plt.subplot(313)
    ppl_percent_change = (loss_mean[1:].exp() - loss_mean[:-1].exp()) / loss_mean[:-1].exp()
    plt.title('Delta PPL % vs seq_len for different fbin')
    cmap = plt.get_cmap('autumn') 
    cNorm  = colors.Normalize(vmin=0, vmax=7)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)
    plt.plot(ppl_percent_change.detach().cpu().numpy()[:, 0], label='Within word', color='b')
    for i in range(8):
        plt.plot(ppl_percent_change.detach().cpu().numpy()[:, i+2], label=f'rarity: {i}', color=scalarMap.to_rgba(i))
    ylim = plt.gca().get_ylim()
    plt.plot(ppl_percent_change.detach().cpu().numpy()[:, 1], label='unknown word', color='k')
    plt.gca().set_ylim(ylim)
    plt.legend()
    return plt.gcf()


def evaluate_longrange(net, ds, n_batches, batch_size,
                       n_seqs, seq_len, min_dist, max_dist,
                       device='cpu', wandb=None, tqdm=None):
    """
    returns
        - loss_mean: (n_seqs, seq_len-1, 11)
        - loss_count: (n_seqs, seq_len-1, 11)
    """
    net.to(device).eval()
    loss_mean = torch.zeros(n_seqs, seq_len-1, 11, device=device)
    loss_count = torch.ones(n_seqs, seq_len-1, 11, device=device)

    pbar = dataset.get_seq_batches(ds, n_batches, batch_size, n_seqs, seq_len,
                                   min_dist, max_dist, unbind=False, device=device)
    pbar = pbar if tqdm is None else tqdm(pbar, total=n_batches)
    for ids, fbins in pbar:
        # ids: (bs, n_seqs, seq_len), fbins: (bs, n_seqs, seq_len)
        with torch.no_grad():
            losses = longrange.loss_fn_longrange(net, ids) #: (bs, n_seqs, seq_len-1)
            loss = losses.mean()

        for i_instance in range(batch_size):
            for i_seq in range(n_seqs):
                f_bins_i = fbins[i_instance, i_seq, 1:] + 2 # +2 to shift to 0-11
                loss_mean_i = loss_mean[i_seq, range(seq_len-1), f_bins_i]
                loss_count_i = loss_count[i_seq, range(seq_len-1), f_bins_i]

                loss_mean[i_seq, range(seq_len-1), f_bins_i] += (losses[i_instance, i_seq] - loss_mean_i)/loss_count_i
                loss_count[i_seq, range(seq_len-1), f_bins_i] += 1

        if tqdm is not None:
            pbar.set_postfix(loss=loss.item())

    return loss_mean.cpu(), loss_count.cpu()
