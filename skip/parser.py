import re
import nltk

import numpy as np
import torch

# TODO rename this file

def book2text(book="../data/murder.txt"):
    with open(book) as f:
        return "".join(f.readlines())
    
def book2ids(book="../data/murder.txt"):
    return text2ids(book2text(book))

def text2ids(text):
    text = text.encode(encoding="ascii", errors="ignore")
    return torch.tensor(list(text), dtype=torch.uint8)

# def get_book_token_ids(book="../data/murder.txt"):
#     with open(book) as f:
#         ids = "".join(f.readlines())
#         ids = [ord(a) for a in list(ids)]
#     return torch.tensor(list(ids))


def ids2text(ids):
    assert ids.ndim>=1
    if ids.ndim==1:
        return "".join([chr(i) for i in ids])
    elif ids.ndim==2:
        return np.array(["".join([chr(i) for i in a]) for a in ids])

def tokenize_word_idx(txt):
    tokens = nltk.tokenize.word_tokenize(txt)
    idxs = []
    offset = 0
    for token in tokens:
        offset = txt.find(token, offset)
        idxs.append(offset)
        offset += len(token)
    tokens, idxs = np.array(tokens), np.array(idxs)
    return tokens, idxs

"""
This method is 20x faster than the one above
"""
def tokenize_word_idx_re(text):
    tokens, idxs = [], []
    # for m in re.finditer('\w+', text): # this includes numbers too; we don't want that
    for m in re.finditer('[a-z]+', text):
        tokens.append(text[m.start(0): m.end(0)])
        idxs.append(m.start(0))
    tokens, idxs = np.array(tokens), np.array(idxs)
    return tokens, idxs

def process_text(text, word2fbin):
    # we only want 32 to 126 inclusive
    possible_chrs = {ord('\n')}.union(set(range(32, 126+1)))
    
    # remove multiple spaces and multiple newlines
    text = re.sub(' +', ' ', text)
    text = re.sub('\n+', '\n', text)
    text = ''.join([a for a in text if ord(a) in possible_chrs])
    
    ids = np.array([ord(a) for a in text], dtype=np.uint8)
    
    words, idx_words = tokenize_word_idx_re(text.lower())
    
    fbin_words = np.array([word2fbin[word] for word in words])
    
    fbin_fchars = np.full(ids.shape, fill_value=-2, dtype=np.int8)
    fbin_fchars[idx_words] = fbin_words
    
    # text, character ids
    # word tokens, locations of word tokens in text
    # frequency bin of each word
    # frequency bin of each (first character in word) and -2 for all other characters
    return dict(text=text, ids=ids, words=words, idx_words=idx_words, fbin_words=fbin_words, fbin_fchars=fbin_fchars)

def calc_fbin2wordset(word2freq, n_bins=9, strategy='uniform'):
    # in a frequency vs word rank chart
    # split evenly on x-axis
    if strategy=='naive':
        fbin2wordset = [set(word_set) for word_set in np.split(words, n_bins)]
    # split evenly on y-axis
    elif strategy=='normal':
        bins = np.logspace(np.log(freqs[0])+1e-5, np.log(freqs[-1])-1e-5, num=n_bins+1, base=np.e)
        fbin2wordset = [{word for word, freq in word2freq.items() if a>=freq and freq>b} for a, b in zip(bins[:-1], bins[1:])]
    # split such that each bin has the same probability
    elif strategy=='uniform':
        binprob = 1./n_bins
        fbin2wordset = []
        wordset, cbinprob = set(), 0.
        for word, freq in word2freq.items():
            wordset.add(word)
            cbinprob += freq
            if cbinprob>=binprob:
                fbin2wordset.append(wordset)
                wordset, cbinprob = set(), 0.
        if wordset:
            fbin2wordset.append(wordset)
    return fbin2wordset

from collections import defaultdict
def calc_word_freq_bins(unigram_freq_csv='../data/unigram_freq.csv', strategy='uniform', tqdm=None):
    word2count = {}
    with open(unigram_freq_csv, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for word, count in reader if tqdm is None else tqdm(reader):
            if count.isnumeric():
                word2count[word.lower()] = int(count)
    n_total_words = np.sum(list(word2count.values()))
    words, wordset = np.array(list(word2count.keys())), set(word2count.keys())
    word2freq = {word: count/n_total_words for word, count in word2count.items()}
    counts, freqs = np.array(list(word2count.values())), np.array(list(word2freq.values()))

    fbin2wordset = calc_fbin2wordset(word2freq, strategy=strategy)
    word2fbin = defaultdict(lambda : -1)
    word2fbin.update({word: fbin for fbin, wordset in enumerate(fbin2wordset) for word in wordset})
    assert sum([len(a) for a in fbin2wordset]) == len(word2freq)
    return word2freq, fbin2wordset, word2fbin



