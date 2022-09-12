import os
import parser

import numpy as np
import torch


def create_dataset(dir_data="../data/Gutenberg/txt", percent_train=0.8, tqdm=None):
    ds = {}
    for file in tqdm(os.listdir(dir_data)):
        file = dir_data + "/" + file
        try:
            ids = parser.book2ids(file)
            if len(ids) > 0:
                ds[file] = ids
        except Exception as e:
            pass

    books_all = sorted(list(ds.keys())).copy()
    np.random.seed(0)
    np.random.shuffle(books_all)
    train_split_idx = int(len(books_all) * percent_train)
    books_train, books_test = books_all[:train_split_idx], books_all[train_split_idx:]

    ds_train, ds_test = {k: ds[k] for k in books_train}, {k: ds[k] for k in books_test}
    return ds_train, ds_test


def get_random_batches(ds, n_batches, batch_size, seq_len):
    books = [book for book, ids in ds.items() if len(ids) > seq_len]
    for idx_batch in range(n_batches):
        batch = []
        while len(batch) < batch_size:
            book = books[np.random.randint(len(books))]
            ids = ds[book]
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

    books = [book for book, ids in ds.items() if len(ids) >= seq_len * 2 + min_dist]
    min_dist, max_dist = min_dist+seq_len, max_dist+seq_len
    # now we only need to worry about the context start indices

    for idx_batch in range(n_batches):
        batch1, batch2 = [], []
        while len(batch1) < batch_size:
            book = books[np.random.randint(len(books))]
            ids = ds[book]
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
