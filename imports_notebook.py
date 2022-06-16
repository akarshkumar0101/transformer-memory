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
from hopfield_memory import HopfieldMemory
from utils import *
import viz

sm = softmax

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
# model = GPT2LMHeadModel.from_pretrained("distilgpt2")