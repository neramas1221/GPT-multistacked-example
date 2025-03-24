from datasets import load_dataset, concatenate_datasets
import pandas as pd
from glob import glob
from torch.utils.data import Dataset
from tqdm import tqdm


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

                for i, val in enumerate(data_set_en):
                    examples.append([val["title"], val["text"], "eng"])
                    if i % 10000 ==0 and i !=0:
                        print(f"Number of examples: {i}")
                        df = pd.DataFrame(examples, columns=["Title", "Text", "Lang"])
                        df.to_csv(f"data/wiki_examples_eng_{int((i/10000))}.csv")
                        examples = [] 
                    if i > max_amount:
                        break

                examples = []

                for i, val in enumerate(data_set_kr):
                    examples.append([val["title"], val["text"], "kor"])
                    if i % 10000 ==0:
                        print(f"Number of examples: {i}")
                        df = pd.DataFrame(examples, columns=["Title", "Text", "Lang"])
                        df.to_csv(f"data/wiki_examples_kor_{int((i/10000))}.csv")
                        examples = []
                    if i > max_amount:
                        break
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
            return self.master_data_set.loc[idx]
        return next(iter(self.master_data_set))


data = WikiData()