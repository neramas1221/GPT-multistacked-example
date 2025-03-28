from datasets import load_dataset, concatenate_datasets
import pandas as pd
from glob import glob
from torch.utils.data import Dataset
from tqdm import tqdm
import re
import tiktoken as tt

class WikiData(Dataset):
    def __init__(self, data_set_name="wikimedia/wikipedia", data_set_chunk="20231101.en", split="train", force_download=False, max_amount=500_000, store=True):
        super().__init__()
        # English Data
        data_set_en = load_dataset(data_set_name, data_set_chunk, split=split, streaming=True)

        # Korean Data
        data_set_kr = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train", streaming=True)
        if store:
            if force_download or len(glob("data/*.csv")) == 0:

                examples = []

                for i, val in enumerate(data_set_en.shuffle()):
                    examples.append([val["title"], val["text"], "eng"])
                    if i % 500 ==0 and i !=0:
                        print(f"Number of examples: {i}")
                        df = pd.DataFrame(examples, columns=["Title", "Text", "Lang"])
                        df.to_csv(f"data/wiki_examples_eng_{int((i/500))}.csv")
                        examples = [] 
                    if i > max_amount:
                        break

                # examples = []

                # for i, val in enumerate(data_set_kr):
                #     examples.append([val["title"], val["text"], "kor"])
                #     if i % 10000 ==0:
                #         print(f"Number of examples: {i}")
                #         df = pd.DataFrame(examples, columns=["Title", "Text", "Lang"])
                #         df.to_csv(f"data/wiki_examples_kor_{int((i/10000))}.csv")
                #         examples = []
                #     if i > max_amount:
                #         break
            data_frames = []
            for f in tqdm(glob("data/*.csv")):
                data_frames.append(pd.read_csv(f))

            self.master_data_set = pd.concat(data_frames)
        else:
            data_set_en = load_dataset(data_set_name, data_set_chunk, split=split, streaming=True)

            # Korean Data
            data_set_kr = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train", streaming=True)
            self.master_data_set = concatenate_datasets([data_set_en, data_set_kr])


    def __len__(self):
        return self.master_data_set.shape[0]


    def __getitem__(self, idx):
        if type(self.master_data_set) == pd.DataFrame:
            return self.master_data_set.iloc[idx]
        return next(iter(self.master_data_set))


def build_encoder(tokens_list):
    # Get all unique tokens
    unique_tokens = set(tokens_list)
    
    # Assign a unique integer ID to each token
    vocab = {token: idx for idx, token in enumerate(unique_tokens, start=4)}  # Start at 4 (0 for padding, 1 for unkown, 2 for start of sentence, 3 for end)
    
    # Add a special token for unknown words (optional)
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    vocab["<STA>"] = 2
    vocab["<END>"] = 3 
    
    return vocab


def encode(vocab, text, max_length=0, generate=False):
    tokens = text
    t = [vocab["<STA>"]]
    t += [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    if generate:
        return t
    if max_length == 0:
        t.append(vocab["<END>"])
        return t
    
    elif len(t) < max_length:
        t.append(vocab["<END>"])
        t += [vocab["<PAD>"] for _ in range(max_length-len(t))]
        return t
    
    elif len(t) > max_length:
        t = t[:max_length]
        t[-1] = vocab["<END>"]
        return t
    
    elif len(t) == max_length:
        t[-1] = vocab["<END>"]
        return t


def decoder(encoder, input):
    decoded_vals = []
    for tar in tqdm(input):
        decoded_vals.append(encoder[tar.tolist()])
    return " ".join(decoded_vals)


def clean_text(text):
    # Define regex pattern to keep only Korean and English words
    pattern = r"[^\w\s'가-힣a-zA-Z]"
    # Remove unwanted characters using regex
    cleaned_text = re.sub(pattern, " ", text)
    # Convert to lowercase
    cleaned_text = cleaned_text.lower()
    # Remove extra whitespace and newlines
    cleaned_text = cleaned_text.strip()
    # Split text into words
    words = cleaned_text.split()
    return words


def create_vocab(data):
    total_vals = set()
    for i, vals in tqdm(enumerate(data)):
        clean_text_set = set(clean_text(vals["Text"]))
        total_vals |= clean_text_set
        if i % 1000 == 0:
            print("\n")
            print(len(total_vals))

    return total_vals


def encode_tick_token(tt_vocab, input):
    return tt_vocab.encode(input)


def decode_tick_token(tt_vocab, input):
    return "".join(tt_vocab.decode(input.tolist()))


def get_tokenizer_tt():
    cl100k_base = tt.get_encoding("cl100k_base")

    enc = tt.Encoding(
        name="cl100k_im",
        pat_str=cl100k_base._pat_str,
        mergeable_ranks=cl100k_base._mergeable_ranks,
        special_tokens={
            **cl100k_base._special_tokens,
            "<|pad_token|>": 100278
        }
    )
    return enc


def encode_with_tt(vocab, text, max_length=512, generate=False):
    """
    special tokens in ticktokenizer cl100k_base model
    {'<|endofprompt|>',
    '<|endoftext|>',
    '<|fim_middle|>',
    '<|fim_prefix|>',
    '<|fim_suffix|>',
    '<|pad_token|>'}
    """
    tokens = text
    t = vocab.encode(tokens, allowed_special=vocab.special_tokens_set)
    if generate:
        return t
    if max_length == 0:
        t.append(100257) # end of string token
        return t
    
    elif len(t) < max_length:
        t.append(100257) # end of string token
        t += [100278 for _ in range(max_length-len(t))] # add the padding tokens
        return t
    
    elif len(t) > max_length:
        t = t[:max_length]
        t[-1] = 100257 # end of string token
        return t
    
    elif len(t) == max_length:
        t[-1] = 100257 # end of string token
        return t
