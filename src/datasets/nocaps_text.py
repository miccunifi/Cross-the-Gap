import json
from collections import defaultdict
from pathlib import Path
from typing import Union

from torch.utils.data import Dataset


class NoCapsTextDataset(Dataset):
    def __init__(self, dataroot: Union[str, Path], split: str):
        dataroot = Path(dataroot)

        assert split in ['val', 'val_gallery', 'val_query'], f'Unknown split: {split}'

        dataset_path = dataroot / 'nocaps'
        self.dataset_path = dataset_path
        self.split = split

        imageid_to_captions = defaultdict(list)
        with open(dataset_path / 'nocaps_val_4500_captions.json', 'r') as f:
            captions = json.load(f)['annotations']
        for caption in captions:
            imageid_to_captions[caption['image_id']].append(caption)

        if 'gallery' in split:  # discard the first caption of each image
            for image_id, captions in imageid_to_captions.items():
                imageid_to_captions[image_id] = captions[1:]
        elif 'query' in split:  # take the first caption of each image as the query
            for image_id, captions in imageid_to_captions.items():
                imageid_to_captions[image_id] = captions[:1]

        image_ids = list(imageid_to_captions.keys())
        self.captions = []
        self.captions_id = []
        self.labels = []
        for image_id, captions in imageid_to_captions.items():
            for caption in captions:
                self.captions.append(caption['caption'])
                self.captions_id.append(caption['id'])
                self.labels.append(image_ids.index(image_id))

    def __getitem__(self, index):
        data = {}
        caption_id = self.captions_id[index]
        caption = self.captions[index]

        data['text'] = caption
        data['text_name'] = str(caption_id)

        return data

    def __len__(self):
        return len(self.captions)

    def get_labels(self, *args, **kwargs):
        if self.split in ['val']:
            raise ValueError('Labels are not available val split')
        return self.labels
