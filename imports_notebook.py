# %load_ext autoreload
# %autoreload 2

import sys

from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

from transformers import AutoTokenizer

from modeling_gpt2 import GPT2Model, GPT2LMHeadModel
from utils import to_np, count_params