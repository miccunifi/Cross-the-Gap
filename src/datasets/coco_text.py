import json
from pathlib import Path
from typing import Union

from torch.utils.data import Dataset


class CocoTextDataset(Dataset):
    def __init__(self, dataroot: Union[str, Path], split: str):
        dataroot = Path(dataroot)

        assert split in ['train', 'train_gallery', 'train_query', 'val', 'val_gallery',
                         'val_query', 'test', 'test_query', 'test_gallery'], f'Unknown split: {split}'

        if 'train' in split:
            filename_split = 'train'
        elif 'val' in split:
            filename_split = 'val'
        elif 'test' in split:
            filename_split = 'test'
        else:
            raise ValueError(f'Unknown split: {split}')

        dataset_path = dataroot / 'COCO2014'
        self.dataset_path = dataset_path
        self.split = split

        with open(dataset_path / 'dataset_coco_light.json', 'r') as f:
            annotations = json.load(f)
        annotations = annotations['images']

        if filename_split == 'train':
            annotations = [annotation for annotation in annotations if annotation['split'] in ['train', 'restval']]
        else:
            annotations = [annotation for annotation in annotations if annotation['split'] == filename_split]

        self.annotations = annotations

        self.captions = []
        self.captions_id = []
        self.labels = []
        for annotation in annotations:
            if 'gallery' in split:
                annotation['sentences'] = annotation['sentences'][1:]
            elif 'query' in split:
                annotation['sentences'] = annotation['sentences'][:1]

            for sentence in annotation['sentences']:
                self.captions.append(sentence['raw'])
                self.captions_id.append(sentence['sentid'])
                self.labels.append(annotation['imgid'])

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
        if self.split in ['train', 'val', 'test']:
            raise ValueError('Labels are not available when "gallery" or "query" are not in the split')
        return self.labels
