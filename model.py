###########################################################################################################
# This is only a model at this stage a needs more work but is the foundation of what will become an LLM bot
# The model is a multi decoder model that takes an input in the form of B,T Batch, Time, and returns B*T, C as the return size
# Where B is batch size (how many examples are shown), T is the time element (The number of tokens shown), C is the number of channels(resulting vocab embedding).
# If no target is given the model will return the logit as B, T,C 
###########################################################################################################

import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np

GPU = 0
# Check what device we are working on, if cuda and GPU is available we will use those
device = torch.device(GPU if torch.cuda.is_available() else 'cpu')

# Veriables
block_size = 256
dropout = 0.1
n_heads = 6 # Number of attention heads
n_emdb = 384 # number of hidden layers
head_size = n_emdb // n_heads
vocab_size = 2000
num_layers = 6 # Number of decoders to make


class Head(nn.Module):
    def __init__(self, n_emdb, hidden_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_emdb, hidden_size)
        self.query = nn.Linear(n_emdb, hidden_size)
        self.value = nn.Linear(n_emdb, hidden_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape

        k = self.key(x)
        q = self.value(x)
        weights = q @ k.transpose(-2, -1) * C **-0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1) # B, T, T
        weights = self.dropout(weights)
        v = self.value(x)
        return weights @ v
    

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_emdb, head_size, block_size) for _ in range(n_head)])
        self.projection = nn.Linear(n_emdb, n_emdb)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out