from typing import List

from torch.utils.data import Dataset


class DummyClassNamesDataset(Dataset):
    def __init__(self, classnames: List[str]):
        self.classnames = classnames

    # For compatibility with other text dataset yield the classname both as text and as text_name
    # During OVI then we will use the classname as name and the text that will be inverted is "a photo of" + classname
    def __getitem__(self, index):
        return {'text': self.classnames[index],
                'text_name': self.classnames[index]
                }

    def __len__(self):
        return len(self.classnames)
