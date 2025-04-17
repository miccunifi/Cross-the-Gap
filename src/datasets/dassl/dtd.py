import os
import random
from pathlib import Path

import PIL
import PIL.Image
from dassl.data.datasets import Datum
from dassl.utils import listdir_nohidden
from torch.utils.data import Dataset

from .oxford_pets import OxfordPets


class DescribableTextures(Dataset):
    dataset_dir = Path("CoOp", "dtd")

    def __init__(self, dataroot: Path, split: str, preprocess: callable):
        super().__init__()
        self.preprocess = preprocess
        self.split = split
        self.dataset_dir = dataroot / self.dataset_dir

        self.images_folder = self.dataset_dir / "images"
        self.labels_path = None
        self.split_path = self.dataset_dir / "split_zhou_DescribableTextures.json"

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.images_folder)
        else:
            # p_tst = 1 - p_trn - p_val
            train, val, test = self.read_and_split_data(self.images_folder, p_trn=0.5, p_val=0.2)
            OxfordPets.save_split(train, val, test, self.split_path, self.images_folder)

        subsample = "all"
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        if self.split == "train":
            self.data = train
        elif self.split == "val":
            self.data = val
        elif self.split == "test":
            self.data = test
        else:
            raise ValueError(f"Invalid split: {self.split}")
        self.labels = [item.label for item in self.data]
        label2classname = {item.label: item.classname for item in self.data}
        self.classnames = [label2classname[label].replace("_", " ") for label in sorted(label2classname)]

    def __getitem__(self, index):
        image_path = str(self.data[index].impath)
        label = self.data[index].label
        image = self.preprocess(PIL.Image.open(image_path))
        image_name = f"{Path(image_path).parent.name}__{Path(image_path).name}"
        return {
            'image': image,
            'image_name': image_name,
            'label': label
        }

    def __len__(self):
        return len(self.data)

    def get_labels(self, *args, **kwargs):
        return self.labels

    def get_classnames(self, *args, **kwargs):
        return self.classnames

    @staticmethod
    def read_and_split_data(image_dir, p_trn=0.5, p_val=0.2, ignored=None, new_cnames=None):
        # The data are supposed to be organized into the following structure
        # =============
        # images/
        #     dog/
        #     cat/
        #     horse/
        # =============
        if ignored is None:
            ignored = []
        categories = listdir_nohidden(image_dir)
        categories = [c for c in categories if c not in ignored]
        categories.sort()

        p_tst = 1 - p_trn - p_val
        print(f"Splitting into {p_trn:.0%} train, {p_val:.0%} val, and {p_tst:.0%} test")

        def _collate(ims, y, c):
            items = []
            for im in ims:
                # is already 0-based
                item = Datum(impath=im, label=y, classname=c)
                items.append(item)
            return items

        train, val, test = [], [], []
        for label, category in enumerate(categories):
            category_dir = os.path.join(image_dir, category)
            images = listdir_nohidden(category_dir)
            images = [os.path.join(category_dir, im) for im in images]
            random.shuffle(images)
            n_total = len(images)
            n_train = round(n_total * p_trn)
            n_val = round(n_total * p_val)
            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0

            if new_cnames is not None and category in new_cnames:
                category = new_cnames[category]

            train.extend(_collate(images[:n_train], label, category))
            val.extend(_collate(images[n_train: n_train + n_val], label, category))
            test.extend(_collate(images[n_train + n_val:], label, category))

        return train, val, test
