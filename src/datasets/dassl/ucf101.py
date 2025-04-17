import os
import re
from pathlib import Path

import PIL
import PIL.Image
from dassl.data.datasets import Datum
from torch.utils.data import Dataset

from .oxford_pets import OxfordPets


class UCF101(Dataset):
    dataset_dir = Path("CoOp", "ucf101")

    def __init__(self, dataroot: Path, split: str, preprocess: callable):
        super().__init__()
        self.preprocess = preprocess
        self.split = split
        self.dataset_dir = dataroot / self.dataset_dir

        self.images_folder = self.dataset_dir / "UCF-101-midframes"
        self.labels_path = None
        self.split_path = self.dataset_dir / "split_zhou_UCF101.json"

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.images_folder)
        else:
            cname2lab = {}
            filepath = os.path.join(self.dataset_dir, "ucfTrainTestlist/classInd.txt")
            with open(filepath, "r") as f:
                lines = f.readlines()
                for line in lines:
                    label, classname = line.strip().split(" ")
                    # conver to 0-based index
                    label = int(label) - 1
                    cname2lab[classname] = label

            trainval = self.read_data(cname2lab, "ucfTrainTestlist/trainlist01.txt")
            test = self.read_data(cname2lab, "ucfTrainTestlist/testlist01.txt")
            train, val = OxfordPets.split_trainval(trainval)
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

    def read_data(self, cname2lab, text_file):
        text_file = os.path.join(self.dataset_dir, text_file)
        items = []

        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")[0]  # trainlist: filename, label
                action, filename = line.split("/")
                label = cname2lab[action]

                elements = re.findall("[A-Z][^A-Z]*", action)
                renamed_action = "_".join(elements)

                filename = filename.replace(".avi", ".jpg")
                impath = os.path.join(self.images_folder, renamed_action, filename)

                item = Datum(impath=impath, label=label, classname=renamed_action)
                items.append(item)

        return items
