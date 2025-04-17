import csv
import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

_SPLITS = {"train": "Ebay_train", "test": "Ebay_test"}

_SUPER_CLASSES = [
    "bicycle",
    "cabinet",
    "chair",
    "coffee_maker",
    "fan",
    "kettle",
    "lamp",
    "mug",
    "sofa",
    "stapler",
    "table",
    "toaster",
]


class StanfordOnlineProducts(Dataset):
    dataset_dir = Path("Stanford_Online_Products")

    def __init__(self, dataroot: Path, split: str, preprocess: callable):
        super().__init__()
        self.preprocess = preprocess

        self.split = split
        if split not in ['train', 'test', 'all']:
            raise ValueError(f"Invalid split: {split}")

        self.dataset_dir = dataroot / self.dataset_dir

        self.data = []
        self.classes = _SUPER_CLASSES
        self.num_classes = len(self.classes)

        # Load data from the appropriate file
        file_paths = [os.path.join(self.dataset_dir, f"{_SPLITS[split]}.txt")] if split in ['train', 'test'] else [
            os.path.join(self.dataset_dir, f"{_SPLITS['train']}.txt"),
            os.path.join(self.dataset_dir, f"{_SPLITS['test']}.txt")]
        for file_path in file_paths:
            with open(file_path, "r") as file_:
                dataset = csv.DictReader(file_, delimiter=" ")
                for row in dataset:
                    self.data.append({
                        "class_id": int(row["class_id"]) - 1,
                        "super_class_id/num": int(row["super_class_id"]) - 1,
                        "super_class_id": self.classes[int(row["super_class_id"]) - 1],
                        "image_path": os.path.join(self.dataset_dir, row["path"]),
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        image_name = sample["image_path"].split('/')[-1].split('.')[0]
        preprocessed_image = self.preprocess(image)

        return {
            "image": preprocessed_image,
            "label": sample["class_id"],
            "image_name": image_name,
            "super_class_id": sample["super_class_id"],
        }

    def get_labels(self, *args, **kwargs):
        return [d['class_id'] for d in self.data]
