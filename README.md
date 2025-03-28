# GPT-Multistacked-example
This is a pure pytorch implementation of a decoder only transformer for article completion using the Wiki Dataset for its training material.

## Implementation
The GPT model consists of the following structure but these can be edited by changing the parameters within the model.py file however retraining will be required to use the model.
- Token Embedding layer (torch Embedding layer)
- Positional Embedding layer (torch Embedding layer)
- Sequental Block of layers made up of the following
  * Multi headed attention block with a Head (consiting of layers for key, query and values as well as an attention mask), projection (Linear layer) and a Dropout layer
  * An MLP model used to compute and understand the output of the Attention blocks (2 Linear layers, ELU activation function and a dropout layer)
  * 2 Layer normalise layers used to normalize the the input to the Self Attention blocks and MLP
- A linear layer of size of the encoder vocabulary that can then be Softmaxed to get the next predicted token in the sequance

## Next steps

* Add streamlit demo to allow users to interact with model
* True GELU activation function and compaire against existing activation function
* Include training graphs in readme
* Give example out put from the bot
* Include video demo of streamlit demo
