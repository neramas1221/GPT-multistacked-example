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

# Creates the attention heads for use in multi headed attention this
class Head(nn.Module):
    """
    A PyTorch module representing a single attention head for use in a transformer-based model.

    This class implements the scaled dot-product attention mechanism, including the calculation
    of queries, keys, and values, and applies a triangular masking mechanism to ensure proper 
    causal attention.

    Args:
        n_emdb (int): Dimension of the input embeddings.
        hidden_size (int): Dimension of the attention head (per-head hidden size).
        block_size (int): Maximum sequence length for the attention mechanism.

    Attributes:
        key (nn.Linear): Linear layer for projecting input to the key space.
        query (nn.Linear): Linear layer for projecting input to the query space.
        value (nn.Linear): Linear layer for projecting input to the value space.
        tril (torch.Tensor): A lower triangular matrix used for masking future tokens in the attention mechanism.
        dropout (nn.Dropout): Dropout layer applied to the attention weights.

    Methods:
        forward(x):
            Computes the attention weights and applies them to the values.
            
            Args:
                x (torch.Tensor): Input tensor of shape (B, T, C), where:
                    - B is the batch size,
                    - T is the sequence length,
                    - C is the embedding dimension.
            
            Returns:
                torch.Tensor: Output tensor of shape (B, T, hidden_size), containing the result 
                of the weighted combination of values.
    """
    def __init__(self, n_emdb, hidden_size, block_size):
        super().__init__()
        # Linear layer is needed for each of the 3 elements K, Q, V
        self.key = nn.Linear(n_emdb, hidden_size)
        self.query = nn.Linear(n_emdb, hidden_size)
        self.value = nn.Linear(n_emdb, hidden_size)
        # This is used to store the tril function from torch which will make a lower trangler matrix of the block size, so if my block size is 16 
        # This will create a 16 x 16 matrix with ones representing a mask of what tokens the model can use.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        # compute the projection of the key value
        k = self.key(x)
        # compute the projection of the query values
        q = self.value(x)
        # the weights are computed using the dot product of q.k however k needs to be resized due to them both being in B, T, C format
        # so need to convert it to be B, C, T format, then the model is multiplies by the square root of the embedding output which is the semantic
        # representation of the input tokens
        weights = q @ k.transpose(-2, -1) * C **-0.5
        # Mask the weights so only the current token and the value before it can be used
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        # convert the weights into softmax to repressent what tokens to be atteneded to any by how much
        weights = F.softmax(weights, dim=-1) # B, T, T
        # apply dropout
        weights = self.dropout(weights)
        # get the projection for the values
        v = self.value(x)
        # perform the dot poduct of the weigth and v to get the weighted sum of the values, the weights show the relarelationship between the keys and the values
        # this will result in emphasing the most relevent values.
        return weights @ v
    

class MultiHeadAttention(nn.Module):
    """
    A PyTorch module implementing multi-head attention, which allows the model
    to focus on different parts of the input sequence simultaneously by using
    multiple attention heads.

    This class aggregates outputs from multiple instances of the `Head` class,
    concatenates them, and projects the result back to the original embedding dimension.

    Args:
        n_head (int): Number of attention heads to use.
        head_size (int): Dimensionality of each attention head.

    Attributes:
        heads (nn.ModuleList): A list of `Head` modules, each representing an attention head.
        projection (nn.Linear): Linear layer for projecting the concatenated attention outputs
                                back to the original embedding dimension.
        dropout (nn.Dropout): Dropout layer applied to the projected outputs.

    Methods:
        forward(x):
            Performs multi-head attention on the input sequence, aggregates the outputs,
            and applies the final projection and dropout.

            Args:
                x (torch.Tensor): Input tensor of shape (B, T, C), where:
                    - B is the batch size,
                    - T is the sequence length,
                    - C is the embedding dimension.

            Returns:
                torch.Tensor: Output tensor of shape (B, T, C), containing the aggregated
                and projected outputs from all attention heads.
    """
    def __init__(self, n_head, head_size):
        super().__init__()
        # Create a number of heads to be used in the attention layer these heads will be used to learn information about the input
        # this means increasing the heads will increase the amount of information the model can learn but increases training time#
        # each head will learn something e.g. one will learn how nouns and adjectives are related, other will learn what the subject 
        # and objects are in the sentance. However the heads will return there outputs in the size of head_size which is n_emdb // n_heads
        # this means that increaing the number of heads in the model too much will result in the information learned by the model being small
        # and resulting in the heads not capturing a meaningful amount of information this can be fixed by increasing the size of n_emdb 
        self.heads = nn.ModuleList([Head(n_emdb, head_size, block_size) for _ in range(n_head)])
        # Learn about the outputs from the attention heads
        self.projection = nn.Linear(n_emdb, n_emdb)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concat the reuslts from the heads back to the original size as the outputs are in the size B, T, head_size so concating them will result
        # in size B T C (head_size * n_heads) = C
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out
    

class MLP(nn.Module):
    """
    A PyTorch module implementing a simple feedforward multi-layer perceptron (MLP),
    used to process input features through two linear transformations with an activation
    function and dropout for regularization.

    Args:
        hidden_size (int): Dimensionality of the input features and output features.
                           Intermediate features are expanded to 4 times this size.

    Attributes:
        mlp (nn.Sequential): A sequential container consisting of:
            - Linear layer to expand input from `hidden_size` to `hidden_size * 4`.
            - ELU activation function for non-linearity.
            - Linear layer to reduce back to `hidden_size`.
            - Dropout layer for regularization.

    Methods:
        forward(x):
            Processes the input tensor through the MLP.

            Args:
                x (torch.Tensor): Input tensor of shape (B, T, hidden_size), where:
                    - B is the batch size,
                    - T is the sequence length,
                    - hidden_size is the feature dimension.

            Returns:
                torch.Tensor: Output tensor of shape (B, T, hidden_size) after
                applying the transformations.
    """
    def __init__(self, hidden_size):
        super().__init__()
        # create a sequential feed forward model that will be used to process the output of the multi headed attention blocks
        # the mlp has its sizes increased here to allow it to learn and understand more information
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*4),
            nn.ELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
            )
    
    def forward(self, x):
        return self.mlp(x)
    

class Blocks(nn.Module):
    """
    A PyTorch module representing a single transformer block, combining 
    multi-head self-attention, a feedforward MLP, and layer normalization. 
    This block forms the building block of transformer architectures.

    Args:
        n_embd (int): Dimensionality of the input embeddings.
        n_heads (int): Number of attention heads in the multi-head self-attention mechanism.

    Attributes:
        sa (MultiHeadAttention): Multi-head self-attention module.
        mlp (MLP): Feedforward multi-layer perceptron module for further processing of the input.
        layernorm1 (nn.LayerNorm): Layer normalization applied before the self-attention module.
        layernorm2 (nn.LayerNorm): Layer normalization applied before the feedforward MLP.

    Methods:
        forward(x):
            Performs the forward pass through the transformer block, applying
            layer normalization, self-attention, and MLP in sequence with residual connections.

            Args:
                x (torch.Tensor): Input tensor of shape (B, T, n_embd), where:
                    - B is the batch size,
                    - T is the sequence length,
                    - n_embd is the embedding dimension.

            Returns:
                torch.Tensor: Output tensor of shape (B, T, n_embd), representing
                the processed input through the transformer block.
    """
    def __init__(self, n_embd, n_heads):
        super().__init__()
        # Here we want to crate the blocks that will be used in the LLM the blocks have the format of the attention block, the feed forward nn
        # and then 2 layer normalises
        head_size = n_emdb // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.mlp = MLP(n_embd)
        # These layernorms are used to incrase performance and computaion time, also help to solve the vanashin/exploding gradiant problem
        self.layernorm1 = nn.LayerNorm(n_embd)
        self.layernorm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.layernorm1(x))
        x = x + self.mlp(self.layernorm2(x))
        return x
    

class LLM(nn.Module):
    """
    A PyTorch module implementing a basic language model (LLM) architecture using
    transformer blocks, including token embeddings, positional embeddings, 
    multiple transformer blocks, and a final linear output layer for predictions.

    This class also includes a `generate` method for autoregressive text generation.

    Args:
        None: The class relies on global variables such as:
            - `vocab_size` (int): Size of the vocabulary.
            - `n_emdb` (int): Dimensionality of the token and positional embeddings.
            - `block_size` (int): Maximum sequence length.
            - `n_heads` (int): Number of attention heads per transformer block.
            - `num_layers` (int): Number of transformer blocks.
            - `dropout` (float): Dropout probability for regularization.

    Attributes:
        token_emb (nn.Embedding): Embedding layer to map input tokens to dense vectors.
        posembeddings (nn.Embedding): Embedding layer to add positional information 
                                      to the token embeddings.
        blocks (nn.Sequential): A sequential container of `Blocks` (transformer blocks).
        output (nn.Linear): A linear layer projecting the final hidden states to vocabulary logits.

    Methods:
        forward(x, y=None):
            Computes the forward pass of the model, processing input tokens through
            embeddings, transformer blocks, and output layers. Optionally computes
            a loss value when target labels `y` are provided.

            Args:
                x (torch.Tensor): Input tensor of token indices, shape (B, T), where:
                    - B is the batch size,
                    - T is the sequence length.
                y (torch.Tensor, optional): Target tensor of token indices, shape (B, T).
                    Defaults to None.

            Returns:
                tuple: A tuple containing:
                    - `logits` (torch.Tensor): Logits tensor of shape (B, T, vocab_size),
                      representing predictions for each token.
                    - `loss` (torch.Tensor or None): Cross-entropy loss if `y` is provided,
                      otherwise None.

        generate(start_x, max_length=1000):
            Autoregressively generates text sequences by predicting one token at a time
            and appending it to the input.

            Args:
                start_x (torch.Tensor): Initial token sequence of shape (B, T), where:
                    - B is the batch size,
                    - T is the starting sequence length.
                max_length (int): Maximum length of the generated sequence. Defaults to 1000.

            Returns:
                torch.Tensor: Tensor of shape (B, max_length), containing the generated
                sequence of token indices.
    """
    def __init__(self):
        super().__init__()
        # Create the embedding dimentions for the model this is always the vocab size by the number of hidden layers
        self.token_emb = nn.Embedding(vocab_size, n_emdb)
        # Size of the sequance of tokes vs hidden layers size so 256 tokens by hidden layer
        self.posembeddings = nn.Embedding(block_size, n_emdb) 
        # Create the blocks for the transformer models
        self.blocks = nn.Sequential(*[Blocks(n_emdb, n_heads) for _ in range(num_layers)])
        self.output = nn.Linear(n_emdb, vocab_size)
    
    def forward(self, x, y=None):
        B, T = x.shape
        # add the represntation of the vocab to the inputs to get the C dimention 
        tok_emb = self.token_emb(x)
        # get the positional embeddings for the input in the T dimention
        pos_emb = self.posembeddings(torch.arange(T, device=device))
        # add the positional embedding to the model tokens
        n_x = tok_emb + pos_emb
        # pass the input to the blocks layers
        x = self.blocks(n_x)
        # compute the logits from the output
        logits = self.output(x)
        # if no target set loss to none
        if y == None:
            loss = None
        else:
            # Get the shap of the logits
            B, T, C = logits.shape
            # convert the logits to be in a 2D format
            logits = logits.view(B*T, C) # make it 2D from 3D
            # torch only accepts the target at shape 
            target = y.view(B*T)
            # calculate the loss
            loss = F.cross_entropy(logits, target)
        # return the results
        return logits, loss

    def generate(self, start_x, max_length=1000):
        # genearte upto the max tokens
        for _ in range(max_length):#
            # ensures that the data is always within the maximum block size
            idx = start_x[:, -block_size:]
            # get the logits
            logits, loss = self(idx)
            # conver the logits in to format B, T, C in to B, C format
            logits = logits[:, -1, :]
            # use the new logits to get a probablity of which token to select
            prob = F.softmax(logits, dim=-1)
            # select the best token
            idx_next = torch.multinomial(prob, num_samples=1)
            # add the new token to the sequance
            idx = torch.concat((idx, idx_next), dim=1)
            # add someway to look for end of sentence chars
        return idx

model = LLM().to(device)

#### JUST FOR TESTING NEED TO ADD REAL DATA ####
x = torch.randint(0,256,(4, 256))
y = torch.randint(0,256,(4, 256))

logits, loss = model(x.to(device),y.to(device))[0]

print(logits.size)