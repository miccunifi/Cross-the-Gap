import pickle
from pathlib import Path
from typing import Union

import pandas as pd
from torch.utils.data import Dataset

from .utils import PROJECT_ROOT


class IMDBTextDataset(Dataset):
    def __init__(self, dataroot: Union[str, Path], split: str):
        dataroot = Path(dataroot)
        assert split in ['all', 'query'], f'Unknown split: {split}'
        self.split = split

        if split in ['all']:
            try:
                with open(PROJECT_ROOT / 'data' / 'summarized_texts' / 'imdb_text' / split /
                        "meta-llama_Llama-3.2-1B-Instruct" / "summarized_text.pkl", 'rb') as f:
                    self.summarized_texts = pickle.load(f)
                print("Summarized texts loaded successfully.")
            except FileNotFoundError:
                print(f"Summarized texts not found for 'imdb_text' {split}")
                self.summarized_texts = None

            dataset_path = dataroot / 'IMDB_Reviews'
            self.dataset_path = dataset_path
            self.split = split

            self.dataframe = pd.read_csv(dataset_path / 'IMDB_Dataset.csv')
            self.sentiments = list(self.dataframe['sentiment'].unique())
            self.labels = [self.sentiments.index(sentiment) for sentiment in self.dataframe['sentiment']]
        else:  # query
            self.split = split
            self.labels = list(range(2))
            self.template_sentences = [
                'a positive review of a movie.', 'a negative review of a movie.']

    def __getitem__(self, index):
        data = {}
        if self.split in ['all']:
            caption = self.dataframe['review'][index]
            label = self.sentiments.index(self.dataframe['sentiment'][index])
            data['long_text'] = caption.strip()
            if self.summarized_texts:
                data['text'] = self.summarized_texts[str(index)]
            data['text_name'] = str(index)
            data['label'] = label
        else:
            data['text'] = self.template_sentences[index]
            data['text_name'] = self.template_sentences[index]
            data['label'] = self.labels[index]
        return data

    def __len__(self):
        return len(self.dataframe) if self.split in ['all'] else len(self.template_sentences)

    def get_labels(self, *args, **kwargs):
        return self.labels
