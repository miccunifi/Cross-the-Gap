import os
from pathlib import Path

import PIL
import PIL.Image
import pandas as pd
from torch.utils.data import Dataset


class CUB(Dataset):
    dataset_dir = Path("CUB_200_2011")

    def __init__(self, dataroot: Path, split: str, preprocess: callable):
        super().__init__()
        self.preprocess = preprocess

        self.split = split
        if split not in ['train', 'test', 'all']:
            raise ValueError(f"Invalid split: {split}")

        self.dataset_dir = dataroot / self.dataset_dir

        self.images_folder = self.dataset_dir / "images"

        self.data = self.load()

    def load(self):
        images = pd.read_csv(self.dataset_dir / "images.txt", sep=' ', names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(self.dataset_dir / "image_class_labels.txt", sep=' ',
                                         names=['img_id', 'target'])
        train_test_split = pd.read_csv(self.dataset_dir / "train_test_split.txt", sep=' ',
                                       names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        data = data.merge(train_test_split, on='img_id')

        if self.split == "train":
            data = data[data.is_training_img == 1]
        elif self.split == "test":
            data = data[data.is_training_img == 0]
        elif self.split == "all":
            pass

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.images_folder / sample.filepath)
        image_name = sample.filepath.split('/')[1].split('.')[0]
        processed_image = self.preprocess(PIL.Image.open(path).convert("RGB"))
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0

        return {
            'image': processed_image,
            'label': target,
            'image_name': image_name
        }

    def get_labels(self, *args, **kwargs):
        return self.data.target.to_list()
