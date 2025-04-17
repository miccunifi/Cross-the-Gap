import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from PIL import ImageFile
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import PROJECT_ROOT, get_dataset, RETRIEVAL_SPLTS
from datasets import DASSL_DATASETS, DummyClassNamesDataset
from retrieval import compute_retrieval
from utils import device, parse_config, parse_command_line_args, merge_configs, load_clip, resume_exp, save_metrics, \
    save_tokens

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.multiprocessing.set_sharing_strategy('file_system')


def ovi_main(args):
    clip_model, clip_model_name, clip_preprocess = load_clip(args.clip_model_name, args.open_clip_pretrained,
                                                             args.use_open_clip, device)
    args.clip_model_name = clip_model_name
    embedding_dim = clip_model.visual_token_embedding_dim

    names_list = []
    global_ovi_pseudo_tokens = torch.empty((0, args.num_pseudo_tokens, embedding_dim))
    losses_log = []

    args.split = args.split if args.split != "" else RETRIEVAL_SPLTS[args.dataset_name]["query"]

    if args.resume_experiment:
        global_ovi_pseudo_tokens, losses_log, names_list = resume_exp(args, method="ovi")

    # Set the experiment path
    experiment_path = (
            PROJECT_ROOT / 'data' / "ovi_pseudo_tokens" / args.dataset_name.lower() / args.split / args.exp_name)

    if experiment_path.exists() and not args.resume_experiment:
        raise ValueError("Training path already exists, please change the experiment name")

    dataset = get_dataset(args.dataroot, args.dataset_name, args.split, clip_preprocess, return_image=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8)

    # Workaround when using classnames as text
    if args.use_class_names:
        assert args.dataset_name in DASSL_DATASETS, "The dataset should be a DASSL dataset"
        classnames = list(set([el.classname for el in dataset.data]))
        dataset = DummyClassNamesDataset(classnames)
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8)

    for batch_idx, batch in enumerate(tqdm(loader, desc='OVI')):
        text = batch.get('text')
        names = batch.get('text_name')
        if names is None:
            names = batch.get('image_name')

        if args.use_class_names and args.template_sentence is not None:
            text = [args.template_sentence.format(c.replace("_", " ")) for c in text]
        else:
            if args.dataset_name in DASSL_DATASETS:
                args.template_sentence = "a photo of {}"  # Default template sentence is "a photo of {}" for DASSL datasets
            else:
                args.template_sentence = "{}"  # Default template sentence is empty for purely textual datasets
            text = [args.template_sentence.format(c.replace("_", " ")) for c in text]

        if np.array(text).ndim == 2:  # Multiple text for each image like in COCO
            text = np.array(text).T.flatten().tolist()
            names = np.array(names).T.flatten().tolist()
            non_empty_indices = [i for i, x in enumerate(text) if x]  # Get the indices of non-empty strings
            text = [text[i] for i in non_empty_indices]
            names = [names[i] for i in non_empty_indices]
            text = clip_model.tokenizer(text).to(device)
            # Note that in this case the actual batch size during OVI will be args.batch_size * number of texts per image
            print(f"Actual batch size during OVI is ")
        else:
            text = clip_model.tokenizer(text).to(device)

        if set(names) <= set(names_list):
            # Skip images that have already been inverted
            continue

        # Extract the text features
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = clip_model.encode_text(text)

        ovi_pseudo_tokens, iteration_losses = ovi(text_features, clip_model,
                                                  learning_rate=args.learning_rate,
                                                  weight_decay=args.weight_decay,
                                                  num_pseudo_tokens=args.num_pseudo_tokens,
                                                  ovi_steps=args.ovi_steps)

        losses_log.append(iteration_losses)
        names_list.extend(names)
        global_ovi_pseudo_tokens = torch.vstack((global_ovi_pseudo_tokens, ovi_pseudo_tokens.detach().cpu()))

        if batch_idx % args.save_frequency == 0 and batch_idx > 0:
            save_tokens(args, experiment_path, global_ovi_pseudo_tokens, names_list, losses_log, method="ovi")

    # Remove images that have been inverted twice (needed when resuming with a different batch size)
    if len(names_list) != len(set(names_list)):
        seen = set()
        unique_idxs = []
        for idx, name in enumerate(names_list):
            if name not in seen:
                seen.add(name)
                unique_idxs.append(idx)
        names_list = [names_list[i] for i in unique_idxs]
        global_ovi_pseudo_tokens = global_ovi_pseudo_tokens[unique_idxs]

    save_tokens(args, experiment_path, global_ovi_pseudo_tokens, names_list, losses_log, method="ovi")

    if args.validate and args.dataset_name not in DASSL_DATASETS:
        args.query_names_list = names_list
        args.query_pseudo_tokens = global_ovi_pseudo_tokens
        print("Validating the OVI pseudo tokens", flush=True)
        metrics = compute_retrieval(query_eval_type="ovi", gallery_eval_type="text", **args)
        print("\nValidation results", flush=True)
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.2f}", flush=True)

        save_metrics(experiment_path, metrics)
    elif args.dataset_name in DASSL_DATASETS:
        print("The dataset is a DASSL dataset, OVI inversion has been done, but no validation is performed", flush=True)


def ovi(text_features, clip_model, learning_rate=2e-2, weight_decay=0.01, num_pseudo_tokens=1, ovi_steps=1000):
    criterion = nn.CosineEmbeddingLoss()
    criterion_target = torch.as_tensor([1], device=device)
    embedding_dim = clip_model.visual_token_embedding_dim
    bs = len(text_features)

    ovi_pseudo_tokens = torch.empty((bs, num_pseudo_tokens, embedding_dim), device=device)
    # Initialize the pseudo tokens with a normal distribution
    nn.init.normal_(ovi_pseudo_tokens, std=0.02)
    ovi_pseudo_tokens = nn.Parameter(ovi_pseudo_tokens)
    # Initialize the optimizer and the scaler
    optimizer = optim.AdamW([ovi_pseudo_tokens], lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler()
    iteration_losses = []
    # Optimize the pseudo tokens for a fixed number of steps
    for _ in range(ovi_steps):
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            ovi_image_features = clip_model.encode_image_with_pseudo_tokens(ovi_pseudo_tokens, num_pseudo_tokens)

            ovi_image_features = F.normalize(ovi_image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            loss = criterion(ovi_image_features, text_features, criterion_target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        iteration_losses.append(loss.detach().cpu().item())
    return ovi_pseudo_tokens, iteration_losses


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Configuration file')
    args, unknown = parser.parse_known_args()
    config = parse_config(args.config)
    # Parse the command-line arguments and merge with the config
    args = parse_command_line_args(config)
    args = merge_configs(config, args)  # Command-line arguments take precedence over config file
    print(args)

    SEED = args.seed
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    np.random.seed(SEED)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    ovi_main(args)


if __name__ == '__main__':
    main()
