import gc
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from clip.model import CLIP
from dotmap import DotMap
from torchmetrics.functional.retrieval import retrieval_average_precision, retrieval_r_precision
from tqdm import tqdm

from data_utils import PROJECT_ROOT, get_dataset, RETRIEVAL_SPLTS
from datasets import DASSL_DATASETS, NANOBEIR_DATASETS
from datasets.roxford_rparis import ROxfordRParisDataset, compute_map
from utils import get_features, load_clip
from utils import retrieval_average_precision_atk

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
base_path = Path(__file__).absolute().parents[1].absolute()


@torch.no_grad()
def compute_retrieval(gallery_eval_type: str, query_eval_type: str, dataroot: Path, dataset_name: str,
                      clip_model_name: str, use_open_clip: bool = False, open_clip_pretrained: str = None, **kwargs):
    clip_model, clip_model_name, clip_preprocess = load_clip(clip_model_name, open_clip_pretrained, use_open_clip,
                                                             device)

    gallery_names_list = kwargs.get('gallery_names_list', None)
    gallery_pseudo_tokens = kwargs.get('gallery_pseudo_tokens', None)
    query_names_list = kwargs.get('query_names_list', None)
    query_pseudo_tokens = kwargs.get('query_pseudo_tokens', None)

    if gallery_pseudo_tokens is not None:
        gallery_pseudo_tokens = gallery_pseudo_tokens.to(device)
    if query_pseudo_tokens is not None:
        query_pseudo_tokens = query_pseudo_tokens.to(device)

    query_split = kwargs.get("query_split", RETRIEVAL_SPLTS[dataset_name]["query"])
    gallery_split = kwargs.get("gallery_split", RETRIEVAL_SPLTS[dataset_name]["gallery"])

    if query_eval_type == 'image' or query_eval_type == 'oti':
        query_features_type = 'image'
        gallery_features_type = 'image'
    elif query_eval_type == 'text' or query_eval_type == 'ovi':
        query_features_type = 'text'
        gallery_features_type = 'text'
    else:
        raise ValueError(f"Unknown eval type {query_eval_type}")

    oti_template_sentence = kwargs.get('oti_template_sentence', 'a photo of {}')

    # Get gallery, query features and labels
    gallery_features, gallery_labels = get_retrieval_features(dataroot, gallery_eval_type, gallery_pseudo_tokens,
                                                              gallery_names_list, clip_model, clip_model_name,
                                                              dataset_name, gallery_split,
                                                              features_type=gallery_features_type,
                                                              use_open_clip=use_open_clip,
                                                              open_clip_pretrained=open_clip_pretrained,
                                                              oti_template_sentence=oti_template_sentence,
                                                              )

    query_features, query_labels = get_retrieval_features(dataroot, query_eval_type, query_pseudo_tokens,
                                                          query_names_list, clip_model, clip_model_name,
                                                          dataset_name, query_split,
                                                          features_type=query_features_type,
                                                          use_open_clip=use_open_clip,
                                                          open_clip_pretrained=open_clip_pretrained,
                                                          oti_template_sentence=oti_template_sentence,
                                                          )

    gallery_features = gallery_features.float().to(device)
    query_features = query_features.float().to(device)

    # Compute the similarity matrices
    gallery_features = F.normalize(gallery_features)
    query_features = F.normalize(query_features)
    similarities = calculate_similarities(dataset_name, gallery_features, query_features, kwargs.get('split_size', 32))

    if query_split == gallery_split and query_features_type == gallery_features_type:
        is_query_gallery_split_same = True
    else:
        is_query_gallery_split_same = False

    metrics = get_retrieval_metrics(dataset_name, similarities, query_labels, gallery_labels,
                                    is_query_gallery_split_same=is_query_gallery_split_same)

    gc.collect()
    torch.cuda.empty_cache()

    # Add the dataset name to each key of the `metrics` dict
    return {f'{dataset_name}_{key}': value for key, value in metrics.items()}


def calculate_similarities(dataset_name: str, gallery_features: torch.Tensor, query_features: torch.Tensor,
                           split_size: Optional[int] = 32):
    # Compute the cosine similarity in a batched manner to avoid memory issues
    if dataset_name == 'imagenet':  # For imagenet we use float16 to save CPU memory
        similarities = torch.empty((query_features.shape[0], gallery_features.shape[0]), device='cpu',
                                   dtype=torch.float16)
        splitted_query_features = torch.split(query_features, split_size)
        with torch.cuda.amp.autocast():
            for i, query_batch in tqdm(enumerate(splitted_query_features), total=len(splitted_query_features),
                                       desc='Computing similarities'):
                start_idx = i * split_size
                end_idx = start_idx + query_batch.size(0)
                similarities[start_idx:end_idx] = torch.matmul(query_batch, gallery_features.T).cpu()
    else:
        similarities = torch.vstack([torch.matmul(query_feat, gallery_features.T).cpu()
                                     for query_feat in
                                     tqdm(query_features.split(split_size), desc="Computing similarities")])
    return similarities


@torch.no_grad()
def get_retrieval_features(dataroot, eval_type: str, pseudo_tokens, names_list: List[str],
                           clip_model: CLIP, clip_model_name: str, dataset_name: str, split: str,
                           use_open_clip: bool = False, open_clip_pretrained: str = None, **kwargs):
    features_type = kwargs.get('features_type', 'image')
    oti_template_sentence = kwargs.get('oti_template_sentence', 'a photo of {}')

    if pseudo_tokens is not None:  # when eval_type is 'image' or 'text'
        num_pseudo_tokens = pseudo_tokens.shape[1]  # (batch_size, num_pseudo_tokens, embedding_dim)

    features, names = get_features(dataroot, dataset_name, split, clip_model_name, features_type,
                                   use_open_clip=use_open_clip, open_clip_pretrained=open_clip_pretrained, **kwargs)

    if eval_type in ['image', 'text']:
        dummy_dataset = get_dataset(dataroot, dataset_name, split, preprocess=lambda x: x,
                                    **kwargs)  # Dataset used only to get the labels
        labels = torch.tensor(dummy_dataset.get_labels(features_type=features_type))

    elif eval_type == 'oti':
        dummy_dataset = get_dataset(dataroot, dataset_name, split, None,
                                    **kwargs)  # Dataset used only to get the labels
        labels = torch.tensor(dummy_dataset.get_labels(features_type=features_type))

        text = clip_model.tokenizer(
            [oti_template_sentence.format(" £ " * num_pseudo_tokens)] * pseudo_tokens.shape[0]).to(device)

        # In encode_with_pseudo_tokens we replace the ' £ ' tokens with the pseudo tokens
        pseudo_tokens = torch.vstack(
            [pseudo_tokens[names_list.index(n)].unsqueeze(0) for n in names_list])  # Make sure the order is right
        splitted_pseudo_tokens = torch.split(pseudo_tokens, 32)
        splitted_text = torch.split(text, 32)
        features = torch.vstack(
            [clip_model.encode_with_pseudo_tokens(batch_text, batch_tokens, num_pseudo_tokens) for
             (batch_text, batch_tokens) in zip(splitted_text, splitted_pseudo_tokens)])

    elif eval_type == 'ovi':
        dummy_dataset = get_dataset(dataroot, dataset_name, split, None,
                                    **kwargs)  # Dataset used only to get the labels
        labels = torch.tensor(dummy_dataset.get_labels(features_type=features_type))

        pseudo_tokens = torch.vstack(
            [pseudo_tokens[names_list.index(n)].unsqueeze(0) for n in names_list])  # Make sure the order is right
        splitted_pseudo_tokens = torch.split(pseudo_tokens, 32)
        features = torch.vstack(
            [clip_model.encode_image_with_pseudo_tokens(batch_tokens, num_pseudo_tokens) for
             batch_tokens in splitted_pseudo_tokens])

    else:
        raise ValueError(f"Unknown eval type {eval_type}")

    return features, labels


def get_retrieval_metrics(dataset_name: str, similarities: torch.Tensor, query_labels: torch.Tensor,
                          gallery_labels: torch.Tensor, is_query_gallery_split_same: bool = False, **kwargs):
    if dataset_name in ['roxford5k', 'rparis6k']:
        return compute_roxford_rparis_metrics(dataset_name, similarities)
    elif dataset_name in ['coco_text', 'flickr30k_text', 'nocaps_text', 'cub2011', 'sop',
                          'newsgroup_text',
                          'imdb_text'] + DASSL_DATASETS + NANOBEIR_DATASETS:  # I think this is the most general case
        return compute_metrics(dataset_name, similarities, query_labels, gallery_labels, is_query_gallery_split_same)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")


def compute_roxford_rparis_metrics(dataset_name: str, similarities: torch.Tensor):
    """
    This function computes the retrieval metrics for the ROxford5k and RParis6k datasets.
    The metrics are computed using the ground truth provided by the ROxfordRParisDataset class.
    The function computes the mean Average Precision (mAP) for three different cases:
    1. Easy: only easy images are considered as relevant
    2. Medium: both easy and hard images are considered as relevant
    3. Hard: only hard images are considered as relevant

    For standardization with the original code, the function uses the following naming convention:
    - mapE: mean Average Precision for easy images
    - mapM: mean Average Precision for medium images
    - mapH: mean Average Precision for hard images
    """

    similarities = similarities.cpu().numpy().T

    gnd = ROxfordRParisDataset.get_ground_truth(dataset_name)

    # ranks = np.argsort(-similarities, axis=1)
    ranks = np.argsort(-similarities, axis=0)

    # evaluate ranks
    ks = [1, 5, 10]

    # search for easy
    gnd_t = []
    for i in range(len(gnd)):  # for each query
        g = {'ok': np.concatenate([gnd[i]['easy']]), 'junk': np.concatenate([gnd[i]['junk'], gnd[i]['hard']])}
        gnd_t.append(g)
    mapE, apsE, mprE, prsE = compute_map(ranks, gnd_t, ks)

    # search for easy & hard
    gnd_t = []
    for i in range(len(gnd)):
        g = {'ok': np.concatenate([gnd[i]['easy'], gnd[i]['hard']]), 'junk': np.concatenate([gnd[i]['junk']])}
        gnd_t.append(g)
    mapM, apsM, mprM, prsM = compute_map(ranks, gnd_t, ks)

    # search for hard
    gnd_t = []
    for i in range(len(gnd)):
        g = {'ok': np.concatenate([gnd[i]['hard']]), 'junk': np.concatenate([gnd[i]['junk'], gnd[i]['easy']])}
        gnd_t.append(g)
    mapH, apsH, mprH, prsH = compute_map(ranks, gnd_t, ks)

    return_dict = {
        'mAP_easy': mapE * 100,
        'mAP_medium': mapM * 100,
        'mAP_hard': mapH * 100
    }
    for idx, k in enumerate(ks):
        return_dict.update({
            f'mP@{k}_easy': mprE[idx] * 100,
            f'mP@{k}_medium': mprM[idx] * 100,
            f'mP@{k}_hard': mprH[idx] * 100
        })

    return return_dict


def compute_metrics(dataset_name: str, similarities: torch.Tensor, query_labels: torch.Tensor,
                    gallery_labels: torch.Tensor, is_query_gallery_split_same: bool = False):
    num_queries, num_gallery = similarities.shape

    if dataset_name in NANOBEIR_DATASETS:
        dummy_dataset = get_dataset('.', dataset_name, 'query', None)
        ground_truth_tensor = dummy_dataset.get_ground_truth()
    else:
        ground_truth_tensor = (query_labels.unsqueeze(1) == gallery_labels.unsqueeze(0))

    aps = []
    aps_at_r = []  # Average precision at R where R is the number of relevant images per query
    precisions_at_r = []
    recall_at_1 = []

    for query in tqdm(range(num_queries), desc='Computing retrieval metrics'):
        query_sim = similarities[query].to(device)
        query_true = ground_truth_tensor[query].to(device)
        relevant_per_query = torch.sum(query_true)

        if is_query_gallery_split_same:  # Remove the query image from the gallery set
            query_sim = torch.cat((query_sim[:query], query_sim[query + 1:]))
            query_true = torch.cat((query_true[:query], query_true[query + 1:]))

        top_5_similarities, top_5_indices = torch.topk(query_sim, 5)

        # mAP
        ap = retrieval_average_precision(query_sim, query_true)
        aps.append(ap.item())

        # mAP at R
        ap_at_r = retrieval_average_precision_atk(query_sim, query_true, top_k=relevant_per_query)
        aps_at_r.append(ap_at_r.item())

        # Precision at R
        precision_at_r = retrieval_r_precision(query_sim, query_true)
        precisions_at_r.append(precision_at_r.item())

        # Recall at 1
        recall_at_1.append(query_true[top_5_indices[0]].int().item())

    return_dict = {
        'mAP': np.mean(aps) * 100,
        'mAP_at_R': np.mean(aps_at_r) * 100,
        'precision_at_R': np.mean(precisions_at_r) * 100,
        'recall_at_1': np.mean(recall_at_1) * 100
    }

    return return_dict


def init_retrieval_args(args: Namespace) -> DotMap:
    if not isinstance(args, DotMap):
        args = DotMap(vars(args), _dynamic=False)

    args.query_split = args.query_split if args.query_split is not None else RETRIEVAL_SPLTS[args.dataset_name]["query"]
    args.gallery_split = args.gallery_split if args.gallery_split is not None else RETRIEVAL_SPLTS[args.dataset_name][
        "gallery"]

    if args.get("clip_model_name") is None and args.get("query_eval_type") in ['image', 'text'] and args.get(
            "gallery_eval_type") in ['image', 'text']:
        raise ValueError("Clip model name is required when using image or text eval type")

    if args.get("query_exp_name") and args.get("query_eval_type") not in ['oti', 'ovi']:
        print(f"Experiment name will not affect results when using {args.get('eval_type')} eval_type")
    if args.get("gallery_exp_name") and args.get("gallery_eval_type") not in ['oti', 'ovi']:
        print(f"Experiment name will not affect results when using {args.get('eval_type')} eval_type")

    clip_model, clip_model_name, _ = load_clip(args.clip_model_name, args.open_clip_pretrained, args.use_open_clip,
                                               device)
    args.clip_model_name = clip_model_name

    query_pseudo_tokens, query_names_list = get_pseudo_tokens(args.dataset_name, args.query_exp_name,
                                                              args.query_eval_type, args.query_split, clip_model_name)
    args.query_pseudo_tokens = query_pseudo_tokens
    args.query_names_list = query_names_list

    gallery_pseudo_tokens, gallery_names_list = get_pseudo_tokens(args.dataset_name, args.gallery_exp_name,
                                                                  args.gallery_eval_type, args.gallery_split,
                                                                  clip_model_name)

    args.gallery_pseudo_tokens = gallery_pseudo_tokens
    args.gallery_names_list = gallery_names_list

    args.use_open_clip = args.get('use_open_clip', False)
    args.open_clip_pretrained = args.get('open_clip_pretrained', None)

    return args


def get_pseudo_tokens(dataset_name, exp_name, eval_type: str, data_split: str, clip_model_name=None):
    if eval_type in ['oti', 'ovi']:
        experiment_path = (PROJECT_ROOT / 'data' / f"{eval_type}_pseudo_tokens" /
                           dataset_name.lower() / data_split / exp_name)
        if not experiment_path.exists():
            raise ValueError(f"Experiment {exp_name} not found")

        with open(experiment_path / 'hyperparameters.json') as f:
            hyperparameters = json.load(f)

        if clip_model_name is not None and clip_model_name != hyperparameters['clip_model_name']:
            raise ValueError(
                f"Clip model name mismatch. Expected {clip_model_name}, found {hyperparameters['clip_model_name']}")

        pseudo_tokens = torch.load(experiment_path / f'{eval_type}_pseudo_tokens.pt', map_location=device)
        with open(experiment_path / 'names.pkl', 'rb') as f:
            names_list = pickle.load(f)
    else:
        pseudo_tokens = None
        names_list = None

    return pseudo_tokens, names_list


def add_args_to_parser():
    parser = ArgumentParser()
    parser.add_argument("--dataroot", required=True, help="path to datasets root")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--clip_model_name", required=True, type=str, help="clip model name, e.g. 'ViT-B/32'")

    parser.add_argument("--query_eval_type", type=str,
                        choices=['oti', 'ovi', 'image', 'text'],
                        required=True, help="If 'oti' evaluate directly using the inverted oti pseudo tokens, "
                                            'if "ovi" evaluate directly using the inverted ovi pseudo tokens,'
                                            "If 'image' use only the image features"
                                            "If 'text' use only the text features")
    parser.add_argument("--gallery_eval_type", type=str,
                        choices=['oti', 'ovi', 'image', 'text'],
                        required=True, help="If 'oti' evaluate directly using the inverted oti pseudo tokens, "
                                            'if "ovi" evaluate directly using the inverted ovi pseudo tokens,'
                                            "If 'image' use only the image features"
                                            "If 'text' use only the text features")

    parser.add_argument("--oti_template_sentence", type=str, default='a photo of {}')

    parser.add_argument("--query_split", type=str, default=None, help="which split to use for queries")
    parser.add_argument("--gallery_split", type=str, default=None, help="which split to use for gallery")

    parser.add_argument("--query_exp_name", type=str, help="experiment name ('phi' or 'oti' to evaluate")
    parser.add_argument("--gallery_exp_name", type=str, help="experiment name ('phi' or 'oti' to evaluate")

    parser.add_argument("--use_open_clip", action='store_true', help="Use OpenCLIP instead of CLIP", default=False)
    parser.add_argument("--open_clip_pretrained", type=str, help="OpenCLIP pretrained model name", default=None)

    return parser


def main():
    parser = add_args_to_parser()
    args = parser.parse_args()
    args = init_retrieval_args(args)
    metrics = compute_retrieval(**args)

    print("\n\n")
    print(f"clip_model_name = {args.clip_model_name}")
    print(f"dataset = {args.dataset_name}")
    print(f"query_eval_type = {args.query_eval_type}")
    print(f"gallery_eval_type = {args.gallery_eval_type}")
    print(f"query_split = {args.query_split}")
    print(f"gallery_split = {args.gallery_split}")
    print(f"query_exp_name = {args.query_exp_name}")
    print(f"gallery_exp_name = {args.gallery_exp_name}")

    print(f"use_open_clip = {args.use_open_clip}")
    print(f"open_clip_pretrained = {args.open_clip_pretrained}")
    print("\n\n")

    results_dict = {}
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name} = {metric_value:.2f}")
        results_dict[metric_name] = metric_value


if __name__ == '__main__':
    main()
