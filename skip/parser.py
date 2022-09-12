import numpy as np
import torch

# TODO rename this file

def book2ids(book="../data/murder.txt"):
    with open(book) as f:
        return text2ids("".join(f.readlines()))

def text2ids(text):
    text = text.encode(encoding="ascii", errors="ignore")
    return torch.tensor(list(text))

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
