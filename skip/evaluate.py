import dataset
import longrange
import torch
from torch import nn


def evaluate_transformer(ds, net, n_batches=10, batch_size=32, n_seqs=4, seq_len=100, device='cpu', wandb=None, tqdm=None):
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    
    net.eval().to(device)
    # average loss
    # average loss across sequence
    # average loss for (first character in word)
    # average loss for (different freq bins)
    
    with torch.no_grad():
        data = {}
        losses = [[] for _ in range(n_seqs)]

        pbar = dataset.get_random_batches_new(ds, n_batches, batch_size, n_seqs, seq_len, min_dist=0, max_dist=1, unbind=True)
        for batch_ids, batch_freqs in tqdm(pbar, total=n_batches):
            memory = None
            for idx_seq, (seq_ids, seq_freqs) in enumerate(zip(batch_ids, batch_freqs)):
                seq_ids, seq_freqs = seq_ids.to(device).long(), seq_freqs.to(device).long()
                logits_i, memory_i = net.forward(seq_ids, memory_in=memory, calc_memory_out=True)
                # print(seq_ids.shape, seq_freqs.shape)
                memory = memory_i if memory is None else torch.cat([memory, memory_i], dim=-2)
                # print(memory.shape)

                a = logits_i[..., :-1, :].reshape(-1, logits_i.shape[-1])
                b = seq_ids[..., 1:].reshape(-1)
                loss = loss_fn(a, b).reshape(batch_size, -1).mean(dim=0).detach().cpu()
                # print(loss.shape)
                losses[idx_seq].append(loss)
        return losses


def evaluate_longrange(ds, net, n_batches=10, batch_size=32, n_seqs=4, seq_len=100, device='cpu', wandb=None, tqdm=None):
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    
    net.eval().to(device)
    
    losses_all = []
    fbin2loss_all = []
    
    pbar = dataset.get_seq_batches(ds=ds, n_batches=n_batches, batch_size=batch_size, n_seqs=n_seqs,
                                   seq_len=seq_len, min_dist=0, max_dist=1, unbind=False,
                                   device=device, dtype=torch.long)
    pbar = tqdm(pbar, total=n_batches) if tqdm else pbar
    
    with torch.no_grad():
        for batch_ids, batch_fbin in pbar:
            losses, fbin2loss = longrange.loss_fn_longrange(net, batch_ids, batch_fbin)
            print(losses.shape, fbin2loss.shape)
            fbin2loss_all.append(fbin2loss)
            losses_all.append(losses.mean(dim=0).detach().cpu())
        losses_all = torch.stack(losses_all)
    fbin2loss_all = {key: torch.stack([fbin2loss[key] for fbin2loss in fbin2loss_all]).mean(dim=0) for key in fbin2loss_all[0].keys()}
    # losses is of shape (n_seq, seq_len-1)
    # fbin2loss_all is dictionary from fbin to array of size n_seq
    return losses_all.mean(dim=0), fbin2loss_all

    