

def evaluate_longrange(ds, net, n_batches=10, batch_size=32, n_seqs=4, seq_len=100, device='cpu', wandb=None, tqdm=None):
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    
    net_longrange.eval()
    net_longrange.to(device)
    
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