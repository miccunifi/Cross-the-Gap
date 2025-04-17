import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).absolute().parents[3].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.data import DataManager

from utils import load_clip

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


# custom implementation
@TRAINER_REGISTRY.register()
class ZeroshotCLIP(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        clip_model, _, _ = load_clip(cfg.MODEL.BACKBONE.NAME,
                                     cfg.MODEL.BACKBONE.OPEN_CLIP_PRETRAINED,
                                     cfg.MODEL.BACKBONE.USE_OPEN_CLIP, self.device)
        clip_model.to(self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip_model.tokenizer(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        _, _, clip_preprocess = load_clip(self.cfg.MODEL.BACKBONE.NAME,
                                          self.cfg.MODEL.BACKBONE.OPEN_CLIP_PRETRAINED,
                                          self.cfg.MODEL.BACKBONE.USE_OPEN_CLIP, 'cpu')
        dm = DataManager(self.cfg, custom_tfm_train=clip_preprocess, custom_tfm_test=clip_preprocess)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm


@TRAINER_REGISTRY.register()
class ZeroShotOTI(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        clip_model, _, _ = load_clip(cfg.MODEL.BACKBONE.NAME,
                                     cfg.MODEL.BACKBONE.OPEN_CLIP_PRETRAINED,
                                     cfg.MODEL.BACKBONE.USE_OPEN_CLIP, self.device)

        custom_temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        self.oti_template = cfg.OTI_TEMPLATE_PROMPT
        print(f"oti_template: {self.oti_template}", flush=True)

        # Load pseudo tokens
        with open(Path(cfg.oti_tokens_path) / "names.pkl", 'rb') as f:
            self.oti_token_names = pickle.load(f)

        self.oti_pseudo_tokens = torch.load(Path(cfg.oti_tokens_path) / 'oti_pseudo_tokens.pt',
                                            map_location='cpu')

        num_tokens = self.oti_pseudo_tokens.shape[1]
        self.num_tokens = num_tokens

        prompts = [custom_temp.format(c.replace("_", " ")) for c in classnames]

        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip_model.tokenizer(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def model_inference(self, name):
        batch_tokens = torch.vstack([self.oti_pseudo_tokens[self.oti_token_names.index(n)].unsqueeze(0) for n in name])
        batch_tokens = batch_tokens.to(self.device)

        template_oti_texts = [self.oti_template.format(" £ " * self.num_tokens) for _ in name]
        template_oti_texts = torch.cat([self.clip_model.tokenizer(p) for p in template_oti_texts])
        template_oti_texts = template_oti_texts.to(self.device)

        oti_features = self.clip_model.encode_with_pseudo_tokens(template_oti_texts, batch_tokens, self.num_tokens)
        oti_features = oti_features / oti_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * oti_features @ self.text_features.t()

        return logits

    def parse_batch_test(self, batch):
        label = batch["label"]
        label = label.to(self.device)

        name = [f"{Path(image_path).parent.name}__{Path(image_path).name}" for image_path in batch['impath']]

        return name, label

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        _, _, clip_preprocess = load_clip(self.cfg.MODEL.BACKBONE.NAME,
                                          self.cfg.MODEL.BACKBONE.OPEN_CLIP_PRETRAINED,
                                          self.cfg.MODEL.BACKBONE.USE_OPEN_CLIP, 'cpu')
        dm = DataManager(self.cfg, custom_tfm_train=clip_preprocess, custom_tfm_test=clip_preprocess)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm


@TRAINER_REGISTRY.register()
class ZeroShotOVI(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        clip_model, _, _ = load_clip(cfg.MODEL.BACKBONE.NAME,
                                     cfg.MODEL.BACKBONE.OPEN_CLIP_PRETRAINED,
                                     cfg.MODEL.BACKBONE.USE_OPEN_CLIP, self.device)

        with open(Path(cfg.ovi_tokens_path) / "names.pkl", 'rb') as f:
            ovi_token_names = pickle.load(f)

        ovi_pseudo_tokens = torch.load(Path(cfg.ovi_tokens_path) / 'ovi_pseudo_tokens.pt',
                                       map_location='cpu')
        num_tokens = ovi_pseudo_tokens.shape[1]

        ovi_pseudo_tokens = torch.vstack(
            [ovi_pseudo_tokens[ovi_token_names.index(n)].unsqueeze(0) for n in classnames])
        ovi_pseudo_tokens = ovi_pseudo_tokens.to(self.device)
        with torch.no_grad():
            ovi_features = clip_model.encode_image_with_pseudo_tokens(ovi_pseudo_tokens, num_tokens)
            ovi_features = ovi_features / ovi_features.norm(dim=-1, keepdim=True)

        self.text_features = ovi_features
        self.clip_model = clip_model

    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        _, _, clip_preprocess = load_clip(self.cfg.MODEL.BACKBONE.NAME,
                                          self.cfg.MODEL.BACKBONE.OPEN_CLIP_PRETRAINED,
                                          self.cfg.MODEL.BACKBONE.USE_OPEN_CLIP, 'cpu')
        dm = DataManager(self.cfg, custom_tfm_train=clip_preprocess, custom_tfm_test=clip_preprocess)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm
