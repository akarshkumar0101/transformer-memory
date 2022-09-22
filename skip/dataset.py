import os
import parser

import numpy as np
import torch

import util


def create_dataset(dir_data="../data/Gutenberg/txt", percent_train=0.8, tqdm=None):
    word2freq, fbin2wordset, word2fbin = parser.calc_word_freq_bins(tqdm=tqdm)
    
    ds = {}
    files = os.listdir(dir_data)
    print(f'Loading {len(files)} books')
    files = [f'{dir_data}/{file}' for file in files]
    
    pbar = tqdm(files) if tqdm is not None else files
    for file in pbar:
        try:
            # ids = parser.book2ids(file)
            book_data = parser.process_text(parser.book2text(file), word2fbin)
            if len(book_data['ids']) > 0:
                ds[file] = book_data
        except Exception as e:
            print(e)

    books_all = sorted(list(ds.keys())).copy()
    np.random.seed(0)
    np.random.shuffle(books_all)
    train_split_idx = int(len(books_all) * percent_train)
    books_train, books_test = books_all[:train_split_idx], books_all[train_split_idx:]

    ds_train, ds_test = {k: ds[k] for k in books_train}, {k: ds[k] for k in books_test}
    return ds_train, ds_test



def load_dataset(tqdm=None):
    if os.path.exists('../data/datasets.pkl'):
        print('Found existing dataset at ../data/datasets.pkl')
        ds_train, ds_test = util.read_object('../data/datasets.pkl', default=(None, None))
    else:
        print('Did NOT find existing dataset at ../data/datasets.pkl, creating new one')
        ds_train, ds_test = create_dataset(tqdm=tqdm)
        util.write_object((ds_train, ds_test), '../data/datasets.pkl')
    return ds_train, ds_test
        


def get_random_batches(ds, n_batches, batch_size, seq_len):
    books = [book for book, data in ds.items() if len(data['ids']) > seq_len]
    for idx_batch in range(n_batches):
        batch = []
        while len(batch) < batch_size:
            book = books[np.random.randint(len(books))]
            ids = torch.from_numpy(ds[book]['ids'])
            # first and last context indices (index of context start)
            cidx_first, cidx_last = 0, len(ids) - seq_len
            i1 = torch.randint(cidx_first, cidx_last + 1, size=(1,)).item()
            batch.append(ids[i1 : i1 + seq_len])
        yield torch.stack(batch)


def get_random_skip_batches(ds, n_batches, batch_size, seq_len, min_dist=None, max_dist=None):
    """
    min_dist (inclusive), max_dist (exclusive)
    """

    if min_dist is None:
        min_dist = 0
    if max_dist is None:
        max_dist = max([len(ids) for ids in ds.values()])
    # print(min_dist, max_dist)
    assert min_dist<=max_dist and min_dist>=0

    books = [book for book, data in ds.items() if len(data['ids']) >= seq_len * 2 + min_dist]
    min_dist, max_dist = min_dist+seq_len, max_dist+seq_len
    # now we only need to worry about the context start indices

    for idx_batch in range(n_batches):
        batch1, batch2 = [], []
        while len(batch1) < batch_size:
            book = books[np.random.randint(len(books))]
            ids = torch.from_numpy(ds[book]['ids'])
            # first and last context indices (index of context start)
            cidx_first, cidx_last = 0, len(ids) - seq_len

            i1 = torch.randint(cidx_first, cidx_last + 1 - min_dist, size=(1,)).item()
            # print('sampling from ', cidx_first, 'to ', cidx_last + 1 - min_dist)
            # print(i1)
            # print('sampling from ', i1 + min_dist, 'to min [', i1+max_dist, cidx_last+1, ']')
            i2 = torch.randint(
                i1 + min_dist,
                min(i1 + max_dist, cidx_last+1),
                size=(1,),
            ).item()
            # print(i2)
            # print()

            batch1.append(ids[i1 : i1 + seq_len])
            batch2.append(ids[i2 : i2 + seq_len])
        yield torch.stack(batch1), torch.stack(batch2)




"""
N tokens
C context length
* * * * * * * * * * * * * * * * *   *
0 1 2 3 4 ...                   N-2 N-1

cidx_first = 0
cidx_last = N-C

"""

def get_random_batches_new(ds, n_batches, batch_size, n_seqs, seq_len, min_dist=None, max_dist=None, unbind=True):
    """
    ds is the dataset
    n_batches in the number of batches to loop over
    batch_size is the batch_size
    n_seqs is the number of seqs (from the same book) to get in a single instance
    seq_len is the length of those sequences
    min_dist is the minimum distance between two seqs (0 means touching)
    max_dist is the maximum distance between two seqs
    
    min_dist (inclusive), max_dist (exclusive)
    unbind decides whether or not iterator should return a tuple of n_seqs
    """
    if min_dist is None:
        min_dist = 0
    if max_dist is None:
        max_dist = max([len(ids) for ids in ds.values()])
    assert min_dist<=max_dist and min_dist>=0

    books = np.array([book for book, data in ds.items() if len(data['ids']) >= seq_len*n_seqs + min_dist * (n_seqs-1)])

    for idx_batch in range(n_batches):
        batch_ids, batch_fbin = [], []
        # batch1, batch2 = [], []
        for book in np.random.choice(books, size=batch_size):
            ids = torch.as_tensor(ds[book]['ids'], dtype=torch.uint8)
            fbin_fchars = torch.as_tensor(ds[book]['fbin_fchars'], dtype=torch.int8)
            
            # create max_dist_book based on book len or whatever max dist is
            max_dist_book = min(max_dist, (len(ids)-seq_len*n_seqs)//(n_seqs-1)+1)
            # min_dist_book = min(
            
            gaps = np.random.randint(low=min_dist, high=max_dist_book, size=n_seqs)
            gaps[0] = 0
            
            total_len = gaps.sum()+seq_len*n_seqs
            
            start = np.random.randint(low=0, high=len(ids)-total_len+1, size=1)
            # 10
            # 5, 3, 2, 4
            # start, start+10+5, start+20+5+3, start+30+5+3+2, start+40+5+3+2+4
            starts = start+gaps.cumsum()+seq_len*np.arange(len(gaps))
            ends = starts+seq_len
            
            batch_ids.append(torch.stack([ids[start: end] for start, end in zip(starts, ends)]))
            batch_fbin.append(torch.stack([fbin_fchars[start: end] for start, end in zip(starts, ends)]))
        batch_ids, batch_fbin = torch.stack(batch_ids), torch.stack(batch_fbin)
        if unbind:
            yield batch_ids.unbind(dim=-2), batch_fbin.unbind(dim=-2)
        else:
            yield batch_ids, batch_fbin

            
 