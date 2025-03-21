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
    

class MLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*4),
            nn.ELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
            )
    
    def forward(self, x):
        return self.mlp(x)
    

class Blocks(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_emdb // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.mlp = MLP(n_embd)
        self.layernorm1 = nn.LayerNorm(n_embd)
        self.layernorm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.layernorm1(x))
        x = x + self.mlp(self.layernorm2(x))
        return x
    

class LLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_emdb)
        self.posembeddings = nn.Embedding(block_size, n_emdb) # Size of the sequance of tokes vs hidden layers size so 256 tokens by hidden layer
        self.blocks = nn.Sequential(*[Blocks(n_emdb, n_heads) for _ in range(num_layers)])
        self.output = nn.Linear(n_emdb, vocab_size)
    
    def forward(self, x, y=None):
        B, T = x.shape
        tok_emb = self.token_emb(x)
        pos_emb = self.posembeddings(torch.arange(T, device=device))
        n_x = tok_emb + pos_emb
        x = self.blocks(n_x)

        logits = self.output(x)

        if y == None:
            loss = None
        else:
            B, T, C = logits.shape
            print(B, T, C)
            logits = logits.view(B*T, C) # make it 2D from 3D
            target = y.view(B*T)
            loss = F.cross_entropy(logits, target)

        return logits, loss

    def generate(self, start_x, max_length=1000):
        for _ in range(max_length):
            idx = start_x[:, -block_size:]
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            prob = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(prob, num_samples=1)
            idx = torch.concat((start_x, idx_next), dim=1)
            # add someway to look for end of sentence chars
        return idx

model = LLM().to(device)