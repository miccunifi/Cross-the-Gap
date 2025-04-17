import sys
from pathlib import Path
from typing import Union

import torch
from torch.utils.data import Dataset

from datasets import *

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(1, str(PROJECT_ROOT))


def get_dataset(dataroot: Union[str, Path], dataset_name: str, split: str, preprocess, **kwargs):
    dataset_name = dataset_name.lower()
    dataroot = Path(dataroot)

    if dataset_name == 'roxford5k':
        dataset = ROxfordRParisDataset(dataroot, 'roxford5k', split, preprocess=preprocess)
    elif dataset_name == 'rparis6k':
        dataset = ROxfordRParisDataset(dataroot, 'rparis6k', split, preprocess=preprocess)
    elif dataset_name == 'cub2011':
        dataset = CUB(dataroot, split, preprocess=preprocess)
    elif dataset_name == 'sop':
        dataset = StanfordOnlineProducts(dataroot, split, preprocess=preprocess)
    elif dataset_name == 'stanford_cars':
        dataset = StanfordCars(dataroot, split, preprocess=preprocess)
    elif dataset_name == 'oxford_pets':
        dataset = OxfordPets(dataroot, split, preprocess=preprocess)
    elif dataset_name == 'oxford_flowers':
        dataset = OxfordFlowers(dataroot, split, preprocess=preprocess)
    elif dataset_name == 'fgvc_aircraft':
        dataset = FGVCAircraft(dataroot, split, preprocess=preprocess)
    elif dataset_name == 'dtd':
        dataset = DescribableTextures(dataroot, split, preprocess=preprocess)
    elif dataset_name == 'eurosat':
        dataset = EuroSAT(dataroot, split, preprocess=preprocess)
    elif dataset_name == 'food101':
        dataset = Food101(dataroot, split, preprocess=preprocess)
    elif dataset_name == 'sun397':
        dataset = SUN397(dataroot, split, preprocess=preprocess)
    elif dataset_name == 'caltech101':
        dataset = Caltech101(dataroot, split, preprocess=preprocess)
    elif dataset_name == 'ucf101':
        dataset = UCF101(dataroot, split, preprocess=preprocess)
    elif dataset_name == 'imagenet':
        dataset = ImageNet(dataroot, split, preprocess=preprocess)
    elif dataset_name == 'coco_text':
        dataset = CocoTextDataset(dataroot, split)
    elif dataset_name == 'flickr30k_text':
        dataset = Flickr30KTextDataset(dataroot, split)
    elif dataset_name == 'nocaps_text':
        dataset = NoCapsTextDataset(dataroot, split)
    elif dataset_name == 'imdb_text':
        dataset = IMDBTextDataset(dataroot, split)
    elif dataset_name == 'newsgroup_text':
        dataset = NewsGroupDataset(split)
    elif dataset_name in ['nanoclimatefever', 'nanodbpedia', 'nanofever',
                          'nanonfcorpus', 'nanonq', 'nanoscidocs',
                          'nanoscifact']:
        dataset = NanoBEIRDataset(split, dataset_name)
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized")
    return dataset


def collate_fn(batch):
    """
    function which discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


RETRIEVAL_SPLTS = {
    # IMAGE DATASETS
    # CUB
    "cub2011": {'query': 'all', 'gallery': 'all'},
    # ROXFORD RPARIS
    "roxford5k": {'query': 'query', 'gallery': 'gallery'},
    "rparis6k": {'query': 'query', 'gallery': 'gallery'},
    # SOP
    "sop": {'query': 'test', 'gallery': 'test'},
    # CALTECH
    "caltech101": {'query': 'test', 'gallery': 'train'},
    # DTD
    "dtd": {'query': 'test', 'gallery': 'train'},
    # EUROSAT
    "eurosat": {'query': 'test', 'gallery': 'train'},
    # AIRCRAFT
    "fgvc_aircraft": {'query': 'test', 'gallery': 'train'},
    # FOOD
    "food101": {'query': 'test', 'gallery': 'train'},
    # IMAGENET
    "imagenet": {'query': 'test', 'gallery': 'train'},
    # OXFORD FLOWERS
    "oxford_flowers": {'query': 'test', 'gallery': 'train'},
    # OXFORD PETS
    "oxford_pets": {'query': 'test', 'gallery': 'train'},
    # CARS
    "stanford_cars": {'query': 'test', 'gallery': 'train'},
    # SUN397
    "sun397": {'query': 'test', 'gallery': 'train'},
    # UCF101
    "ucf101": {'query': 'test', 'gallery': 'train'},
    # TEXT DATASETS
    # COCO TEXT
    "coco_text": {'query': 'test_query', 'gallery': 'test_gallery'},
    # FLICKR 30K TEXT
    "flickr30k_text": {'query': 'val_query', 'gallery': 'val_gallery'},
    # IMDB TEXT
    "imdb_text": {'query': 'query', 'gallery': 'all'},
    # CLIMATE FEVER
    "nanoclimatefever": {'query': 'query', 'gallery': 'gallery'},
    # DBPEDIA
    "nanodbpedia": {'query': 'query', 'gallery': 'gallery'},
    # FEVER
    "nanofever": {'query': 'query', 'gallery': 'gallery'},
    # NFCORPUS
    "nanonfcorpus": {'query': 'query', 'gallery': 'gallery'},
    # NANO NQ
    "nanonq": {'query': 'query', 'gallery': 'gallery'},
    # SCIDOCS
    "nanoscidocs": {'query': 'query', 'gallery': 'gallery'},
    # SCIFACT
    "nanoscifact": {'query': 'query', 'gallery': 'gallery'},
    # NEWSGROUP TEXT
    "newsgroup_text": {'query': 'query', 'gallery': 'test'},
    # NOCAPS TEXT
    "nocaps_text": {'query': 'val_query', 'gallery': 'val_gallery'},
}

CLASSIFICATION_SPLTS = {
    # CALTECH
    "caltech101": {'split': 'test'},
    # DTD
    "dtd": {'split': 'test'},
    # EUROSAT
    "eurosat": {'split': 'test'},
    # AIRCRAFT
    "fgvc_aircraft": {'split': 'test'},
    # FOOD
    "food101": {'split': 'test'},
    # IMAGENET
    "imagenet": {'split': 'val'},
    # OXFORD FLOWERS
    "oxford_flowers": {'split': 'test'},
    # OXFORD PETS
    "oxford_pets": {'split': 'test'},
    # CARS
    "stanford_cars": {'split': 'test'},
    # SUN397
    "sun397": {'split': 'test'},
    # UCF101
    "ucf101": {'split': 'test'},
}
