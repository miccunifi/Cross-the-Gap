import pickle

from sklearn.datasets import fetch_20newsgroups
from torch.utils.data import Dataset

from .utils import PROJECT_ROOT

class_to_template = {
    'alt.atheism': 'Atheism, philosophy, and the absence of belief in deities.',
    'comp.graphics': 'Computer graphics, rendering, and visual technologies.',
    'comp.os.ms-windows.misc': 'Microsoft Windows: software, settings, and troubleshooting.',
    'comp.sys.ibm.pc.hardware': 'IBM PC hardware: components, peripherals, and systems.',
    'comp.sys.mac.hardware': 'Mac hardware: devices, accessories, and configurations.',
    'comp.windows.x': 'X Windows: graphical interfaces and Unix system configuration.',
    'misc.forsale': 'Buying and selling products like electronics and furniture.',
    'rec.autos': 'Automobiles: car models, maintenance, and industry updates.',
    'rec.motorcycles': 'Motorcycles: bike models, culture, and maintenance tips.',
    'rec.sport.baseball': 'Baseball: games, teams, players, and statistics.',
    'rec.sport.hockey': 'Hockey: NHL, teams, games, and techniques.',
    'sci.crypt': 'Cryptography: encryption, security, and data protection.',
    'sci.electronics': 'Electronics: circuits, components, and device technology.',
    'sci.med': 'Medicine: healthcare, treatments, and medical research.',
    'sci.space': 'Space: exploration, astronomy, and scientific discoveries.',
    'soc.religion.christian': 'Christianity: beliefs, theology, and practices.',
    'talk.politics.guns': 'Gun politics: control, rights, and related issues.',
    'talk.politics.mideast': 'Middle East politics: conflicts, diplomacy, and events.',
    'talk.politics.misc': 'Politics: issues, governance, and international relations.',
    'talk.religion.misc': 'Religion and spirituality: diverse beliefs and philosophies.',
}


class NewsGroupDataset(Dataset):
    def __init__(self, split: str):
        # dataroot = Path(dataroot)
        assert split in ['train', 'test', 'all', 'query'], f'Unknown split: {split}'
        self.split = split

        if split in ['train', 'test', 'all']:
            self.dataset = fetch_20newsgroups(subset=split)
            try:
                with open(PROJECT_ROOT / 'data' / 'summarized_texts' / 'newsgroup_text' / split /
                          "meta-llama_Llama-3.2-1B-Instruct" / "summarized_text.pkl", 'rb') as f:
                    self.summarized_texts = pickle.load(f)
                print("Summarized texts loaded successfully.")
            except FileNotFoundError:
                print(f"Summarized texts not found for 'newsgroup_text' {split}")
                self.summarized_texts = None
            self.labels = self.dataset.target
        else:
            dummy_dataset = fetch_20newsgroups(subset='train')
            self.target_names = dummy_dataset.target_names
            self.class_to_template = class_to_template
            self.labels = [self.target_names.index(class_name) for class_name in self.class_to_template]

    def __getitem__(self, index):
        data = {}
        if self.split in ['train', 'test', 'all']:
            caption = self.dataset.data[index]
            label = self.dataset.target[index]
            data['long_text'] = caption.strip()
            if self.summarized_texts:
                data['text'] = self.summarized_texts[str(index)]
            data['text_name'] = str(index)
            data['label'] = label
            return data
        else:
            data['text'] = self.class_to_template[self.target_names[index]]
            data['text_name'] = self.target_names[index]
            data['label'] = self.labels[index]
            return data

    def __len__(self):
        if self.split in ['train', 'test', 'all']:
            return len(self.dataset.data)
        else:
            return len(self.class_to_template)

    def get_labels(self, *args, **kwargs):
        return self.labels
