import os
from pathlib import Path

import PIL
import PIL.Image
from dassl.data.datasets import Datum
from torch.utils.data import Dataset

from .oxford_pets import OxfordPets


class FGVCAircraft(Dataset):
    dataset_dir = Path("CoOp", "fgvc_aircraft")

    def __init__(self, dataroot: Path, split: str, preprocess: callable):
        super().__init__()
        self.preprocess = preprocess
        self.split = split
        self.dataset_dir = dataroot / self.dataset_dir

        self.images_folder = self.dataset_dir / "images"
        self.labels_path = None
        self.split_path = None

        classnames = []
        with open(os.path.join(self.dataset_dir, "variants.txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                classnames.append(line.strip())
        cname2lab = {c: i for i, c in enumerate(classnames)}

        train = self.read_data(cname2lab, "images_variant_train.txt")
        val = self.read_data(cname2lab, "images_variant_val.txt")
        test = self.read_data(cname2lab, "images_variant_test.txt")

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

    def read_data(self, cname2lab, split_file):
        filepath = os.path.join(self.dataset_dir, split_file)
        items = []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                imname = line[0] + ".jpg"
                classname = " ".join(line[1:])
                impath = os.path.join(self.images_folder, imname)
                label = cname2lab[classname]
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items
