import pickle

import numpy as np


def read_object(filename, default=None):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except:
        return default
        
def write_object(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

        
def count_params(net):
    return np.sum([p.numel() for p in net.parameters()], dtype=int)



