import os
import pickle
import shutil
from pathlib import Path

import PIL.Image
import numpy as np
from PIL import ImageFile
from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import PROJECT_ROOT

ImageFile.LOAD_TRUNCATED_IMAGES = True


def configdataset(dataset, dir_main):
    DATASETS = ['roxford5k', 'rparis6k', 'revisitop1m']

    dataset = dataset.lower()

    if dataset not in DATASETS:
        raise ValueError('Unknown dataset: {}!'.format(dataset))

    if dataset == 'roxford5k' or dataset == 'rparis6k':
        # loading imlist, qimlist, and gnd, in cfg as a dict
        gnd_fname = os.path.join(dir_main, dataset, 'gnd_{}.pkl'.format(dataset))
        with open(gnd_fname, 'rb') as f:
            cfg = pickle.load(f)
        cfg['gnd_fname'] = gnd_fname
        cfg['ext'] = '.jpg'
        cfg['qext'] = '.jpg'

    elif dataset == 'revisitop1m':
        # loading imlist from a .txt file
        cfg = {}
        cfg['imlist_fname'] = os.path.join(dir_main, dataset, '{}.txt'.format(dataset))
        cfg['imlist'] = read_imlist(cfg['imlist_fname'])
        cfg['qimlist'] = []
        cfg['ext'] = ''
        cfg['qext'] = ''

    cfg['dir_data'] = os.path.join(dir_main, dataset)
    cfg['dir_images'] = os.path.join(cfg['dir_data'], 'jpg')

    cfg['n'] = len(cfg['imlist'])
    cfg['nq'] = len(cfg['qimlist'])

    cfg['im_fname'] = config_imname
    cfg['qim_fname'] = config_qimname

    cfg['dataset'] = dataset

    return cfg


def config_imname(cfg, i):
    return os.path.join(cfg['dir_images'], cfg['imlist'][i] + cfg['ext'])


def config_qimname(cfg, i):
    return os.path.join(cfg['dir_images'], cfg['qimlist'][i] + cfg['qext'])


def read_imlist(imlist_fn):
    with open(imlist_fn, 'r') as file:
        imlist = file.read().splitlines()
    return imlist


def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.

    Arguments
    ---------
    ranks : zero-based ranks of positive images
    nres  : number of positive images

    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap


def compute_map(ranks, gnd, kappas=None):
    """
    Computes the mAP for a given set of returned results.

         Usage:
           map = compute_map (ranks, gnd)
                 computes mean average precsion (map) only

           map, aps, pr, prs = compute_map (ranks, gnd, kappas)
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query

         Notes:
         1) ranks starts from 0, ranks.shape = db_size X number_of_queries
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """

    if kappas is None:
        kappas = []
    map = 0.
    nq = len(gnd)  # number of queries
    aps = np.zeros(nq)  # average precision for each query
    pr = np.zeros(len(kappas))  # precision at kappas
    prs = np.zeros((nq, len(kappas)))  # precision at kappas for each query
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.array(gnd[i]['ok'])

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]['junk'])
        except:
            qgndj = np.empty(0)

        # sorted positions of positive and junk images (0 based)
        pos = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgndj)]

        k = 0
        ij = 0
        if len(junk):
            # decrease positions of positives based on the number of
            # junk images appearing before them
            ip = 0
            while ip < len(pos):
                while ij < len(junk) and pos[ip] > junk[ij]:
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        # compute precision @ k
        pos += 1  # get it to 1-based
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j])
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    map = map / (nq - nempty)
    pr = pr / (nq - nempty)

    return map, aps, pr, prs


class ROxfordRParisDataset(Dataset):
    def __init__(self, dataroot: Path, dataset: str, split: str, preprocess: callable):
        super().__init__()
        if dataset not in ['roxford5k', 'rparis6k']:
            raise ValueError("Dataset should be `roxford5k` of `rparis6k`")

        if split not in ['gallery', 'query']:
            raise ValueError("Split should be 'gallery' or 'query' ")

        self.preprocess = preprocess
        self.dataset = dataset
        self.split = split

        self.cfg = configdataset(dataset, dataroot)
        if split == "gallery":
            self.cfg_distractors = configdataset('revisitop1m', dataroot)

        if split == 'query':
            self.image_paths = [Path(self.cfg['dir_images']) / (x + ".jpg") for x in self.cfg['qimlist']]
            self.image_names = self.cfg['qimlist']
        else:
            self.image_paths = [Path(self.cfg['dir_images']) / (x + ".jpg") for x in self.cfg['imlist']]
            self.image_names = self.cfg['imlist']
            self.find_existing_distractors()

        if not (PROJECT_ROOT / 'data' / 'roxford_rparis_gnds' / f'gnd_{dataset}.pkl').exists():
            (PROJECT_ROOT / 'data' / 'roxford_rparis_gnds').mkdir(exist_ok=True, parents=True)
            shutil.copy(self.cfg['gnd_fname'], PROJECT_ROOT / 'data' / 'roxford_rparis_gnds' / f'gnd_{dataset}.pkl')

        # print(f"Dataset: {dataset}, Split: {split}, Number of images: {len(self.image_paths)}")

        # self.cfg = None Questo ci serve per gnd
        self.cfg_distractors = None
        self.labels = [-1] * len(self.image_paths)

    def find_existing_distractors(self):
        images_path = self.cfg_distractors['dir_images']
        for subdir in tqdm(os.listdir(images_path), desc="Finding existing distractors"):
            for image in os.listdir(Path(images_path) / subdir):
                self.image_paths.append(Path(images_path) / subdir / image)
                self.image_names.append(image)

    def __getitem__(self, index):
        image = PIL.Image.open(self.image_paths[index])
        processed_image = self.preprocess(image)
        image_name = self.image_names[index]
        label = self.labels[index]

        return {'image': processed_image,
                'image_name': image_name,
                'label': label
                }

    def __len__(self):
        return len(self.image_paths)

    def get_labels(self, *args, **kwargs):
        return self.labels

    @staticmethod
    def get_ground_truth(dataset_name):
        with open(PROJECT_ROOT / 'data' / 'roxford_rparis_gnds' / f'gnd_{dataset_name}.pkl', 'rb') as f:
            gnd = pickle.load(f)
        return gnd
