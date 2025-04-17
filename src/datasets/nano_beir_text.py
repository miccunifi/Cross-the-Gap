import pickle

import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import PROJECT_ROOT

lower_to_name = {
    'nanoclimatefever': 'NanoClimateFEVER',
    'nanodbpedia': 'NanoDBPedia',
    'nanofever': 'NanoFEVER',
    'nanonfcorpus': 'NanoNFCorpus',
    'nanonq': 'NanoNQ',
    'nanoscidocs': 'NanoSCIDOCS',
    'nanoscifact': 'NanoSciFact'
}


class NanoBEIRDataset(Dataset):
    def __init__(self, split: str, name: str):

        assert name in ['nanoclimatefever', 'nanodbpedia', 'nanofever', 'nanonfcorpus', 'nanonq',
                        'nanoscidocs', 'nanoscifact'], f'Unknown dataset: {name}'

        upper_name = lower_to_name[name]
        assert split in ['query', 'gallery'], f'Unknown split: {split}'
        self.split = split
        self.name = name

        self.gallery = pd.read_parquet(f"hf://datasets/zeta-alpha-ai/{upper_name}/corpus/train-00000-of-00001.parquet")
        self.qrels = pd.read_parquet(f"hf://datasets/zeta-alpha-ai/{upper_name}/qrels/train-00000-of-00001.parquet")
        self.queries = pd.read_parquet(f"hf://datasets/zeta-alpha-ai/{upper_name}/queries/train-00000-of-00001.parquet")

        if split == 'gallery':
            try:
                with open(PROJECT_ROOT / 'data' / 'summarized_texts' / lower_to_name[name] / split /
                        "meta-llama_Llama-3.2-1B-Instruct" / "summarized_text.pkl", 'rb') as f:
                    self.summarized_texts = pickle.load(f)
                print("Summarized texts loaded successfully.")
            except FileNotFoundError:
                print(f"Summarized texts not found for {name} {split}")
                self.summarized_texts = None

        if split == 'query':
            self.dataframe = self.queries
        else:
            self.dataframe = self.gallery

    def __getitem__(self, index):
        data = {}
        if self.split == 'gallery':
            caption = self.dataframe['text'][index]
            data['long_text'] = caption.strip()
            if self.summarized_texts:
                data['text'] = self.summarized_texts[self.dataframe['_id'][index]]
            data['text_name'] = self.dataframe['_id'][index]
        else:
            data['text'] = self.dataframe['text'][index]
            data['text_name'] = self.dataframe['_id'][index]
        return data

    def __len__(self):
        return len(self.dataframe)

    def get_labels(self, *args, **kwargs):  # Fake labels as in ROxfordRParisDataset
        return [0] * len(self.dataframe)

    def get_ground_truth(self):
        ground_truth_tensor = torch.zeros(len(self.queries), len(self.gallery))
        for i in range(len(self.queries)):
            gallery_ids = self.qrels[self.qrels['query-id'] == self.queries['_id'][i]]['corpus-id']
            positive_idx = self.gallery[self.gallery['_id'].isin(gallery_ids)].index
            ground_truth_tensor[i, positive_idx] = 1

        return ground_truth_tensor.int()
