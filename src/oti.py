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
from retrieval import compute_retrieval
from utils import device, parse_config, parse_command_line_args, merge_configs, load_clip, resume_exp, save_metrics, \
    save_tokens

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.multiprocessing.set_sharing_strategy('file_system')


def oti_main(args):
    clip_model, clip_model_name, clip_preprocess = load_clip(args.clip_model_name, args.open_clip_pretrained,
                                                             args.use_open_clip, device)
    args.clip_model_name = clip_model_name
    embedding_dim = clip_model.text_token_embedding_dim

    names_list = []
    global_oti_pseudo_tokens = torch.empty((0, args.num_pseudo_tokens, embedding_dim))
    losses_log = []

    args.split = args.split if args.split != "" else RETRIEVAL_SPLTS[args.dataset_name]["query"]

    if args.resume_experiment:
        global_oti_pseudo_tokens, losses_log, names_list = resume_exp(args, method="oti")

    # Set the experiment path
    experiment_path = (
            PROJECT_ROOT / 'data' / "oti_pseudo_tokens" / args.dataset_name.lower() / args.split / args.exp_name)

    if experiment_path.exists() and not args.resume_experiment:
        raise ValueError(
            "Training path already exists, please change the experiment name or set resume_experiment to True")

    dataset = get_dataset(args.dataroot, args.dataset_name, args.split, clip_preprocess)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8)

    template_sentence = args.template_sentence

    for batch_idx, batch in enumerate(tqdm(loader, desc='OTI')):
        images = batch.get('image')
        names = batch.get('image_name')

        if set(names) <= set(names_list):
            # Skip images that have already been inverted
            continue

        # Extract the image features
        with torch.no_grad(), torch.cuda.amp.autocast():
            batch_im_features = clip_model.encode_image(images.to(device))

        oti_pseudo_tokens, iteration_losses = oti(batch_im_features, clip_model,
                                                  learning_rate=args.learning_rate,
                                                  weight_decay=args.weight_decay,
                                                  num_pseudo_tokens=args.num_pseudo_tokens,
                                                  oti_steps=args.oti_steps,
                                                  template_sentence=args.template_sentence)

        losses_log.append(iteration_losses)
        names_list.extend(names)
        global_oti_pseudo_tokens = torch.vstack((global_oti_pseudo_tokens, oti_pseudo_tokens.detach().cpu()))

        if batch_idx % args.save_frequency == 0 and batch_idx > 0:
            save_tokens(args, experiment_path, global_oti_pseudo_tokens, names_list, losses_log, method="oti")

    # Remove images that have been inverted twice (needed when resuming with a different batch size)
    if len(names_list) != len(set(names_list)):
        seen = set()
        unique_idxs = []
        for idx, name in enumerate(names_list):
            if name not in seen:
                seen.add(name)
                unique_idxs.append(idx)
        names_list = [names_list[i] for i in unique_idxs]
        global_oti_pseudo_tokens = global_oti_pseudo_tokens[unique_idxs]

    save_tokens(args, experiment_path, global_oti_pseudo_tokens, names_list, losses_log, method="oti")

    if args.validate:
        args.query_names_list = names_list
        args.query_pseudo_tokens = global_oti_pseudo_tokens
        print("Validating the OTI pseudo tokens", flush=True)
        metrics = compute_retrieval(query_eval_type="oti", gallery_eval_type="image", **args)
        print("\nValidation results", flush=True)
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.2f}", flush=True)

        save_metrics(experiment_path, metrics)


def oti(batch_im_features, clip_model, learning_rate=2e-2, weight_decay=0.01, num_pseudo_tokens=1, oti_steps=150,
        template_sentence='a photo of {}'):
    criterion = nn.CosineEmbeddingLoss()
    criterion_target = torch.as_tensor([1], device=device)
    embedding_dim = clip_model.text_token_embedding_dim
    bs = len(batch_im_features)

    oti_pseudo_tokens = torch.empty((bs, num_pseudo_tokens, embedding_dim), device=device)
    # Initialize the pseudo tokens with a normal distribution
    nn.init.normal_(oti_pseudo_tokens, std=0.02)
    oti_pseudo_tokens = nn.Parameter(oti_pseudo_tokens)
    # Initialize the optimizer and the scaler
    optimizer = optim.AdamW([oti_pseudo_tokens], lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler()
    iteration_losses = []
    # Optimize the pseudo tokens for a fixed number of steps
    for _ in range(oti_steps):
        optimizer.zero_grad()
        # a photo of £
        template_oti_texts = [template_sentence.format(" £ " * num_pseudo_tokens) for _ in range(bs)]

        tokenized_template_oti_texts = clip_model.tokenizer(template_oti_texts).to(device)

        with torch.cuda.amp.autocast():
            template_oti_texts_features = clip_model.encode_with_pseudo_tokens(tokenized_template_oti_texts,
                                                                               oti_pseudo_tokens,
                                                                               num_pseudo_tokens)

            batch_im_features = F.normalize(batch_im_features, dim=-1)
            template_oti_texts_features = F.normalize(template_oti_texts_features, dim=-1)

            loss = criterion(template_oti_texts_features, batch_im_features, criterion_target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        iteration_losses.append(loss.detach().cpu().item())
    return oti_pseudo_tokens, iteration_losses


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
    print(f"Fixing the seed to {SEED}, for reproducibility")
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    np.random.seed(SEED)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    oti_main(args)


if __name__ == '__main__':
    main()
