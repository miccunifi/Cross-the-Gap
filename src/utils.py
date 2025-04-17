import argparse
import json
import multiprocessing
import pickle
from distutils.util import strtobool
from functools import partial
from functools import reduce
from operator import getitem
from typing import Optional, Literal

import clip
import numpy as np
import open_clip.transformer
import torch
import yaml
from clip.model import CLIP
from dotmap import DotMap
from torch.utils.data import DataLoader
from torchmetrics.utilities.checks import _check_retrieval_functional_inputs
from tqdm import tqdm

from SLIP import SimpleTokenizer as SLIPSimpleTokenizer
from SLIP import load_slip
from data_utils import PROJECT_ROOT
from data_utils import collate_fn, get_dataset
from encode_with_pseudo_tokens import get_encode_with_pseudo_tokens, get_encode_image_with_pseudo_tokens

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

torch.multiprocessing.set_sharing_strategy('file_system')


@torch.no_grad()
def get_features(dataroot, dataset_name: str, split: str, clip_model_name, type: Literal['image', 'text'],
                 use_open_clip=False, open_clip_pretrained=None, **kwargs):
    """
    Return the desired image features, if they are already extracted, otherwise extract them and save them.
    """
    assert type in ['image', 'text'], f"Type must be either 'image' or 'text', but got {type}"

    features_path = (PROJECT_ROOT / 'data' / f'{type}_features' / dataset_name /
                     clip_model_name.replace('/', ''))

    features_path = features_path / split

    if (features_path / f'{type}_features.pt').exists():
        features = torch.load(features_path / f'{type}_features.pt', map_location='cpu')
        with open(features_path / f'{type}_names.pkl', 'rb') as f:
            names = pickle.load(f)
        print(f"Features already extracted {dataset_name} - {split}  - {clip_model_name}")
        return features.float(), names
    else:
        print(f"Extracting features {dataset_name} - {split} - {clip_model_name}")
        features_path.mkdir(parents=True, exist_ok=True)

        clip_model, clip_model_name, clip_preprocess = load_clip(clip_model_name, open_clip_pretrained, use_open_clip,
                                                                 device)

        dataset = get_dataset(dataroot, dataset_name, split, clip_preprocess, **kwargs)

        if type == 'image':
            features, names = extract_image_features(dataset, clip_model)
        elif type == 'text':
            features, names = extract_text_features(dataset, clip_model)
        else:
            raise ValueError(f"Unknown type: {type}")

        torch.save(features, features_path / f"{type}_features.pt")
        with open(features_path / f'{type}_names.pkl', 'wb+') as f:
            pickle.dump(names, f)

        return features.float(), names


@torch.no_grad()
def extract_text_features(dataset, clip_model, batch_size=32):
    loader = DataLoader(dataset=dataset, batch_size=batch_size,
                        num_workers=min(multiprocessing.cpu_count(), 32), pin_memory=True, collate_fn=collate_fn)

    text_features = []
    text_names = []
    try:
        print(f"extracting text features {dataset.__class__.__name__} - {dataset.split}")
    except Exception as e:
        pass

    # Extract features
    for batch in tqdm(loader, desc=f"Extracting text features"):
        texts = batch.get('text')
        names = batch.get('text_name')
        if names is None:
            names = batch.get('image_name', batch.get('name'))

        if np.array(texts).ndim == 2:  # Multiple text for each image like in COCO
            texts = np.array(texts).T.flatten().tolist()
            names = np.array(names).T.flatten().tolist()
            non_empty_indices = [i for i, x in enumerate(texts) if x]  # Get the indices of non-empty strings
            texts = [texts[i] for i in non_empty_indices]
            names = [names[i] for i in non_empty_indices]
            texts = clip_model.tokenizer(texts).to(device)
        else:
            texts = clip_model.tokenizer(texts).to(device)

        batch_features = clip_model.encode_text(texts)

        text_features.append(batch_features.cpu())
        text_names.extend(names)

    text_features = torch.vstack(text_features)
    return text_features, text_names


@torch.no_grad()
def extract_image_features(dataset, clip_model, batch_size=32):
    loader = DataLoader(dataset=dataset, batch_size=batch_size,
                        num_workers=min(multiprocessing.cpu_count(), 32), pin_memory=True, collate_fn=collate_fn)

    image_features = []
    image_names = []
    try:
        print(f"extracting image features {dataset.__class__.__name__} - {dataset.split}")
    except Exception as e:
        pass

    # Extract features
    for batch in tqdm(loader, desc=f"Extracting image features"):
        images = batch.get('image')
        names = batch.get('image_name')

        images = images.to(device)
        batch_features = clip_model.encode_image(images)

        image_features.append(batch_features.cpu())
        image_names.extend(names)

    image_features = torch.vstack(image_features)
    return image_features, image_names


def retrieval_average_precision_atk(preds: torch.Tensor, target: torch.Tensor,
                                    top_k: Optional[int] = None) -> torch.Tensor:
    preds, target = _check_retrieval_functional_inputs(preds, target)

    top_k = top_k or preds.shape[-1]
    if not isinstance(top_k, int) and top_k <= 0:
        raise ValueError(f"Argument ``top_k`` has to be a positive integer or None, but got {top_k}.")

    number_of_relevant = target.sum()
    sorted_indices = torch.topk(preds, k=top_k).indices
    target = target[sorted_indices]
    precisions = torch.cumsum(target, dim=0) * target  # Consider only positions corresponding to GTs
    precisions = precisions / torch.arange(1, precisions.shape[0] + 1,
                                           device=device)  # Compute precision for each position

    return torch.sum(precisions) / min(number_of_relevant, top_k)


def load_clip(clip_model_name, open_clip_pretrained, use_open_clip, local_device):
    if "SLIP" in clip_model_name:
        print("Loading SLIP model: ", clip_model_name)
        clip_model, clip_preprocess = load_slip(clip_model_name, device=local_device)
        clip_model = clip_model.to(local_device)

        # Add the tokenizer to the model
        tokenizer = SLIPSimpleTokenizer()
        clip_model.tokenizer = tokenizer

        clip_model.dtype = next(clip_model.parameters()).dtype

        # Add some needed attributes
        clip_model.visual.input_resolution = clip_model.visual.patch_embed.img_size[0]
        clip_model.text_token_embedding_dim = clip_model.token_embedding.embedding_dim
        clip_model.visual_token_embedding_dim = clip_model.visual.embed_dim
        clip_model.visual.output_dim = clip_model.image_projection.shape[1]


    elif use_open_clip:
        if open_clip_pretrained in clip_model_name:
            clip_model_name = clip_model_name.replace(f"-{open_clip_pretrained}", "")
        import open_clip
        print("Loading OpenCLIP model: ", clip_model_name, " with pretrained: ", open_clip_pretrained)
        clip_model_name = clip_model_name.replace('/', '-')
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(clip_model_name,
                                                                               pretrained=open_clip_pretrained)
        clip_model = clip_model.to(local_device)
        # Add the tokenizer to the model
        tokenizer = open_clip.get_tokenizer(clip_model_name)
        clip_model.tokenizer = tokenizer

        clip_model.dtype = next(clip_model.parameters()).dtype

        # Add some needed attributes
        clip_model.eval()
        clip_model.visual.input_resolution = clip_model.visual.image_size[0]
        clip_model_name = clip_model_name + "-" + open_clip_pretrained

        if isinstance(clip_model, open_clip.CLIP):
            clip_model.text_token_embedding_dim = clip_model.token_embedding.embedding_dim
        elif isinstance(clip_model, open_clip.CustomTextCLIP):
            clip_model.text_token_embedding_dim = clip_model.text.token_embedding.embedding_dim

        if isinstance(clip_model.visual, open_clip.transformer.VisionTransformer):
            clip_model.visual_token_embedding_dim = clip_model.visual.class_embedding.shape[0]
        elif isinstance(clip_model.visual, open_clip.timm_model.TimmModel):
            clip_model.visual_token_embedding_dim = clip_model.visual.trunk.embed_dim

        if isinstance(clip_model.visual, open_clip.timm_model.TimmModel):
            clip_model.visual.output_dim = clip_model.visual.trunk.num_features
    else:  # Load the original CLIP model
        print("Loading OpenAI CLIP model: ", clip_model_name)
        clip_model, clip_preprocess = clip.load(clip_model_name, device=local_device)
        clip_model = clip_model.to(local_device)

        # Add the tokenizer to the model
        tokenizer = partial(clip.tokenize, truncate=True)
        clip_model.tokenizer = tokenizer

        clip_model.text_token_embedding_dim = clip_model.token_embedding.embedding_dim
        if isinstance(clip_model.visual, clip.model.VisionTransformer):
            clip_model.visual_token_embedding_dim = clip_model.visual.class_embedding.shape[0]

    clip_model: CLIP = clip_model.float()
    clip_model.requires_grad_(False)
    # Add the encode_with_pseudo_tokens methods to the model
    encode_with_pseudo_tokens = get_encode_with_pseudo_tokens(clip_model)
    clip_model.encode_with_pseudo_tokens = partial(encode_with_pseudo_tokens, clip_model)

    encode_image_with_pseudo_tokens = get_encode_image_with_pseudo_tokens(clip_model)
    clip_model.encode_image_with_pseudo_tokens = partial(encode_image_with_pseudo_tokens, clip_model)

    return clip_model, clip_model_name, clip_preprocess


def parse_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return DotMap(config, _dynamic=False)


def parse_command_line_args(config):
    parser = argparse.ArgumentParser()

    # Automatically add command-line arguments based on the config structure
    def add_arguments(section, prefix=''):
        for key, value in section.items():
            full_key = f'{prefix}.{key}' if prefix else key
            if isinstance(value, dict):
                add_arguments(value, prefix=full_key)
            else:
                if isinstance(value, list):
                    # Convert list to comma-separated string
                    if not value:
                        parser.add_argument(f'--{full_key}', default=value, nargs='+',
                                            help=f'Value for {full_key}')
                    else:
                        parser.add_argument(f'--{full_key}', default=value, type=type(value[0]), nargs='+',
                                            help=f'Value for {full_key}')
                else:
                    if type(value) == bool:
                        parser.add_argument(f'--{full_key}', default=value, type=strtobool,
                                            help=f'Value for {full_key}')
                    else:
                        parser.add_argument(f'--{full_key}', default=value, type=type(value),
                                            help=f'Value for {full_key}')

    add_arguments(config)

    args, _ = parser.parse_known_args()
    args = DotMap(vars(args), _dynamic=False)
    return args


def merge_configs(config, args):
    keys_to_modify = []

    def update_config(config, key, value):
        *keys, last_key = key.split('.')
        reduce(getitem, keys, config)[last_key] = value

    # Recursively merge command-line parameters into the config
    def get_updates(section, args, prefix=''):
        for key, value in section.items():
            full_key = f'{prefix}.{key}' if prefix else key
            if isinstance(value, dict):
                get_updates(value, args, prefix=full_key)
            elif getattr(args, full_key, None) or getattr(args, full_key, None) != getattr(section, key, None):
                keys_to_modify.append((full_key, getattr(args, full_key)))

    get_updates(config, args)

    for key, value in keys_to_modify:
        update_config(config, key, value)

    return config


def resume_exp(args, method):
    print("Resuming training from a saved experiment", flush=True)
    # Load names and pseudo tokens
    with open(PROJECT_ROOT / "data" / f"{method}_pseudo_tokens" / args.dataset_name.lower() / args.split /
              args.exp_name / f"names.pkl", 'rb') as f:
        names_list = pickle.load(f)
    # load the losses_log
    with open(PROJECT_ROOT / "data" / f"{method}_pseudo_tokens" / args.dataset_name.lower() / args.split /
              args.exp_name / f'losses.pkl', 'rb') as f:
        losses_log = pickle.load(f)
    global_pseudo_tokens = torch.load(
        PROJECT_ROOT / "data" / f"{method}_pseudo_tokens" / args.dataset_name.lower() / args.split /
        args.exp_name / f'{method}_pseudo_tokens.pt')
    # Load the saved hyperparameters
    with open(PROJECT_ROOT / "data" / f"{method}_pseudo_tokens" / args.dataset_name.lower() / args.split /
              args.exp_name / 'hyperparameters.json') as f:
        old_hyperparamters = json.load(f)
    # Check if the hyperparameters are the same
    for k, v in old_hyperparamters.items():
        if k in args:
            if v != args[k]:
                print(f"Warning: {k} is different from the saved experiment")
                print(f"saved parameter: {v} \t new_parameter: {args[k]}")
    return global_pseudo_tokens, losses_log, names_list


def save_metrics(experiment_path, metrics):
    with open(experiment_path / 'metrics.json', 'w+') as f:
        json.dump(metrics, f, sort_keys=True, indent=4)


def save_tokens(args, experiment_path, global_pseudo_tokens, names_list, losses_log, method):
    experiment_path.mkdir(exist_ok=True, parents=True)

    with open(experiment_path / f'names.pkl', 'wb+') as f:
        pickle.dump(names_list, f)
    with open(experiment_path / f'losses.pkl', 'wb+') as f:
        pickle.dump(losses_log, f)
    torch.save(global_pseudo_tokens, experiment_path / f'{method}_pseudo_tokens.pt')
    with open(experiment_path / 'hyperparameters.json', 'w+') as f:
        json.dump(args, f, sort_keys=True, indent=4)
