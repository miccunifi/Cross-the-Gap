from collections import defaultdict
from pathlib import Path
from typing import Union

import pandas as pd
from torch.utils.data import Dataset


# note that in results.csv downloaded from kaggle the images 2199200615.jpg lacks a delimiter
class Flickr30KTextDataset(Dataset):
    def __init__(self, dataroot: Union[str, Path], split: str):
        dataroot = Path(dataroot)

        dataset_path = dataroot / 'Flickr30K'
        self.dataset_path = dataset_path
        self.split = split

        assert split in ['train', 'train_query', 'train_gallery', 'val', 'val_query', 'val_gallery', 'test',
                         'test_query',
                         'test_gallery', 'all', 'all_query', 'all_gallery']
        if 'train' in split:
            filename_split = 'train'
        elif 'val' in split:
            filename_split = 'val'
        elif 'test' in split:
            filename_split = 'test'
        elif 'all' in split:
            filename_split = 'all'
        else:
            raise ValueError(f'Unknown split: {split}')

        split_image_names = None
        if filename_split != 'all':
            with open(dataset_path / f'karpathy_{filename_split}.txt', 'r') as f:
                split_image_names = set(f.read().splitlines())  # These are without the .jpg extension

        self.image_to_captions = defaultdict(list)
        annotation_csv = pd.read_csv(dataset_path / 'results.csv', delimiter='|')
        # iterate over the rows of the csv file
        for index, row in annotation_csv.iterrows():
            if split_image_names and row['image_name'].replace('.jpg', '') not in split_image_names:
                continue
            augmented_caption = {'id': row[' comment_number'], 'caption': row[' comment']}
            self.image_to_captions[row['image_name']].append(augmented_caption)

        if 'gallery' in split:  # discard the first caption of each image
            for image_name, captions in self.image_to_captions.items():
                self.image_to_captions[image_name] = captions[1:]
        elif 'query' in split:  # take the first caption of each image as the query
            for image_name, captions in self.image_to_captions.items():
                self.image_to_captions[image_name] = captions[:1]

        self.image_names = list(self.image_to_captions.keys())
        self.captions = []
        self.caption_names = []
        self.labels = []
        for image_name, captions in self.image_to_captions.items():
            for caption in captions:
                self.captions.append(caption['caption'])
                self.caption_names.append(f"{image_name.strip()}_{caption['id']}")
                self.labels.append(self.image_names.index(image_name))

    def __getitem__(self, index):
        data = {}
        caption = self.captions[index]
        data['text'] = caption.strip()
        data['text_name'] = self.caption_names[index]
        return data

    def __len__(self):
        return len(self.captions)

    def get_labels(self, *args, **kwargs):
        if self.split in ['train', 'val', 'test', 'all']:
            raise ValueError('Labels are not available for the train and val splits')
        return self.labels
