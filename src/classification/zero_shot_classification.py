import argparse
import sys
from pathlib import Path

import torch
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from dassl.utils import set_random_seed, setup_logger

PROJECT_ROOT = Path(__file__).absolute().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from utils import parse_config
from data_utils import CLASSIFICATION_SPLTS

import classification.datasets.caltech101
import classification.datasets.dtd
import classification.datasets.eurosat
import classification.datasets.fgvc_aircraft
import classification.datasets.food101
import classification.datasets.imagenet
import classification.datasets.oxford_flowers
import classification.datasets.oxford_pets
import classification.datasets.stanford_cars
import classification.datasets.sun397
import classification.datasets.ucf101

import classification.trainers

lower_to_name = {
    "caltech101": 'Caltech101',
    'dtd': 'DescribableTextures',
    'eurosat': 'EuroSAT',
    'fgvc_aircraft': 'FGVCAircraft',
    'food101': 'Food101',
    'imagenet': 'ImageNet',
    'oxford_flowers': 'OxfordFlowers',
    'oxford_pets': 'OxfordPets',
    'stanford_cars': 'StanfordCars',
    'sun397': 'SUN397',
    'ucf101': 'UCF101',
}


def update_cfg(cfg, args):
    if args.dataroot:
        cfg.DATASET.ROOT = args.dataroot

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.seed:
        cfg.SEED = args.seed

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    cfg.DATASET.NAME = lower_to_name[args.dataset_name]
    cfg.MODEL.BACKBONE.NAME = args.clip_model_name
    cfg.MODEL.BACKBONE.OPEN_CLIP_PRETRAINED = args.open_clip_pretrained
    cfg.MODEL.BACKBONE.USE_OPEN_CLIP = args.use_open_clip

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    cfg.EXP_NAME = args.exp_name  # default is empty string

    if args.oti_template_prompt == "":
        cfg.OTI_TEMPLATE_PROMPT = "a photo of {}"
    else:
        cfg.OTI_TEMPLATE_PROMPT = args.oti_template_prompt


def setup_cfg(args):
    cfg = get_cfg_default()

    update_cfg(cfg, args)

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    dataset_name = args.dataset_name

    split = CLASSIFICATION_SPLTS[dataset_name]['split']

    if args.eval_type in ["oti", "ovi"]:
        if args.eval_type == 'oti':
            oti_exp_path = PROJECT_ROOT / "data" / "oti_pseudo_tokens" / dataset_name / split / cfg.EXP_NAME
            cfg.oti_tokens_path = str(oti_exp_path)

        if args.eval_type == 'ovi':
            ovi_exp_path = PROJECT_ROOT / "data" / "ovi_pseudo_tokens" / dataset_name / split / cfg.EXP_NAME
            cfg.ovi_tokens_path = str(ovi_exp_path)

        cfg.TRAINER.NAME = "ZeroShotOTI" if args.eval_type == 'oti' else "ZeroShotOVI"
    else:
        cfg.TRAINER.NAME = "ZeroshotCLIP"

    trainer = build_trainer(cfg)
    trainer.test(split=split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default="", help="path to dataset", required=True)
    parser.add_argument("--output-dir", type=str, default="", help="output directory", required=True)
    parser.add_argument("--dataset_name", type=str, help="name of dataset", required=True)
    parser.add_argument('--config', type=str, help='Configuration file', required=True)
    parser.add_argument("--seed", type=int, default=1, help="only positive value enables a fixed seed")
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--eval-type", type=str, choices=['oti', 'ovi', 'zeroshot'], default='zeroshot')

    # OTI and OVI SPECIFIC
    parser.add_argument("--exp-name", type=str, help="OTI/OVI experiment name", default="")

    args = parser.parse_args()

    config = parse_config(args.config)

    args.clip_model_name = config["clip_model_name"]
    args.use_open_clip = config["use_open_clip"]
    args.open_clip_pretrained = config["open_clip_pretrained"]
    args.oti_template_prompt = config["oti_template_prompt"]

    print(args)

    main(args)
