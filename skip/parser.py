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

def process_text(text):
    # we only want 32 to 126 inclusive
    possible_chrs = {ord('\n')}.union(set(range(32, 126+1)))
    
    # remove multiple spaces and multiple newlines
    text = re.sub(' +', ' ', text)
    text = re.sub('\n+', '\n', text)
    text = ''.join([a for a in text if ord(a) in possible_chrs])
    
    ids = np.array([ord(a) for a in text], dtype=np.uint8)
    
    words, idx_words = tokenize_word_idx(text.lower())
    
    fbin_words = np.array([word2fbin[word] for word in words])
    
    fbin_fchars = np.full(ids.shape, fill_value=-2, dtype=np.int8)
    fbin_fchars[idx_words] = fbins_words
    
    # text, character ids
    # word tokens, locations of word tokens in text
    # frequency bin of each word
    # frequency bin of each (first character in word) and -2 for all other characters
    return dict(text=text, ids=ids, words=words, idx_words=idx_words, fbin_words=fbin_words, fbin_fchars=fbin_fchars)