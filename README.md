# Cross the Gap (ICLR 2025)
### Exposing the Intra-modal Misalignment in CLIP via Modality Inversion

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2502.04263)
[![OpenReview](https://img.shields.io/badge/ICLR-Paper-red.svg)](https://openreview.net/forum?id=VVVfuIcmKR)
[![ICLR video](https://img.shields.io/badge/ICLR-Video-red.svg)](https://iclr.cc/virtual/2025/poster/29411)
[![Slides](https://img.shields.io/badge/Slides-Link-orange.svg)](/assets/CrossTheGap_SLIDES.pdf)
[![Poster](https://img.shields.io/badge/Poster-Link-purple.svg)](/assets/CrossTheGap_POSTER.pdf)
[![GitHub Stars](https://img.shields.io/github/stars/miccunifi/Cross-the-Gap?style=social)](https://github.com/miccunifi/Cross-the-Gap)

🔥🔥 **[2025/04/23] Our code is available! Feel free to explore, use, and contribute!** 🔥🔥

This is the **official repository** of the [**ICLR 2025 paper**](https://arxiv.org/abs/2502.04263) 
"*Cross the Gap: Exposing the Intra-modal Misalignment in CLIP via Modality Inversion*" 
by Marco Mistretta*, Alberto Baldrati*, Lorenzo Agnolucci*, Marco Bertini and Andrew David Bagdanov.

Here you can find the implementation of the following modality inversion techniques:

- *Optimization-based Textual Inversion* ([OTI](https://github.com/miccunifi/Cross-the-Gap?tab=readme-ov-file#oti-minimal-working-example)): from visual features &rarr; to textual features
- *Optimization-based Visual Inversion* ([OVI](https://github.com/miccunifi/Cross-the-Gap?tab=readme-ov-file#ovi-minimal-working-example)): from textual features &rarr; to visual features
  
that empirically show the impact of **intra-modal misalignment** in contrastively trained VLMs!



## Overview

### Abstract
Pre-trained multi-modal Vision-Language Models like CLIP are widely used off-the-shelf for various applications.  
In this paper, we show that the common practice of individually exploiting the text or image encoders of these powerful multimodal models is highly suboptimal for intra-modal tasks like image-to-image retrieval.
We argue that this is inherently due to the CLIP-style inter-modal contrastive loss, which does not enforce any intra-modal constraints, leading to what we call **intra-modal misalignment**. To demonstrate this, we leverage two optimization-based **modality inversion** techniques that map representations from their input modality to the complementary one without any need for auxiliary data or additional trained adapters.
We empirically show that, in the intra-modal tasks of image-to-image and text-to-text retrieval, approaching these tasks inter-modally significantly improves performance compared to intra-modal baselines on more than fifteen datasets.
Additionally, we demonstrate that approaching a native inter-modal task (e.g., zero-shot image classification) **intra-modally** decreases performance, further validating our findings. Lastly, we show that incorporating an intra-modal term in the pre-training objective or narrowing the modality gap between the text and image feature embedding spaces helps reduce the intra-modal misalignment.

![assets/teaser.png](assets/teaser.png "Teaser of the method")

***Left***: The inter-modal contrastive loss used in pretraining enforces paired images and texts to be at a given distance $r$ (e.g., $r_{\text{dog}}$ and $r_{\text{cat}}$) but does not encourage **intra-modal** alignment. Consequently, intra-modal similarity scores might not correspond to those of actual images and texts (e.g., $d_1 < d_2$).
***Right***: We show that the common practice of individually exploiting only one encoder is suboptimal. Approaching intra-modal tasks (e.g., image-to-image retrieval) **inter-modally** via **modality inversion** improves performance.

## Citation
```bibtex
@inproceedings{mistretta2025cross,
  title={Cross the Gap: Exposing the Intra-modal Misalignment in CLIP via Modality Inversion},
  author={Marco Mistretta and Alberto Baldrati and Lorenzo Agnolucci and Marco Bertini and Andrew D. Bagdanov},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=VVVfuIcmKR}
}
```

<details>
<summary><h2>Installation Guide</h2></summary> 

This guide provides step-by-step instructions on how to set up the **cross-the-gap** conda environment and install all necessary dependencies. The codebase has been tested on **Ubuntu 20.04.2 LTS** with **Python 3.10**.

1. Create and Activate Conda Environment
```bash
conda create -y -n cross-the-gap python=3.10
conda activate cross-the-gap
```

2. Ensure you have the correct version of *PyTorch* and *torchvision*
```bash
# CUDA 12.1
conda install pytorch==2.1.1 torchvision==0.16.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

3. Cloning Cross-the-Gap and Installing Dassl Library and Requirements
```bash
git clone https://github.com/miccunifi/Cross-the-Gap.git
cd Cross-the-Gap/
pip install git+https://github.com/KaiyangZhou/Dassl.pytorch
chmod +x install_requirements.sh
./install_requirements.sh
```

</details>

<details>
<summary><h2>Dataset Installation Guide</h2></summary> 

Our code currently supports **25+ datasets** for the tasks of image-to-image retrieval, text-to-text retrieval, and zero-shot classification.
Classification datasets have been downloaded following the [CoOp Dataset Preparation Guide](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md).

A Dataset Preparation Guide will be made as soon as possible for the remaining datasets.

To **add a new dataset**, create a `<dataset_name>.py` file in the `src/datasets/` directory with the following methods:

- **Method 1: Initialization (`__init__`)**  
  Initializes dataset directories and reads splits into `self.data`, storing labels in `self.labels` and class names in `self.classnames`.  
  `__init__` implementation should respect the following signature:  
  ```python
  def __init__(self, dataroot: Path, split: str, preprocess: callable):
      """Initialize dataset, read data splits, and store labels and class names."""
      # preprocesses using input preprocess
      self.preprocess = preprocess

      # your code implementation
  
      # Select dataset split based on input argument "split"
      self.data = ...

      # Extract labels and class names
      self.labels = ...
      self.classnames = ...
  ```

- **Method 2: Image Loading (`__getitem__`)**  
  Loads and preprocesses an image, returning a dictionary with the image, name, and label.  
  Example:  
  ```python
  def __getitem__(self, index):
      """Load and preprocess an image, returning image, name, and label."""

     # your code implementation
  
     return {
          'image': image,            # Processed image
          'image_name': image_name,  # Image name
          'label': label             # Ground-truth label (if needed) or dummy label
      }
  ```

- **Method 3: Dataset Length (`__len__`)**  
  Returns the dataset size.
 
- **Method 4: Getters for Labels and Class Names**  
  ```python
  def get_labels(self):
      """Return the list of all labels."""
      return self.labels

  def get_classnames(self):
      """Return the list of all class names."""
      return self.classnames
  ```

Implementing the above methods you can easily add your custom dataset. Remember to set a dataset name and referencing it in `src/data_utils.py` and in `src/datasets/__init__.py`  

</details>

<details>
<summary><h2>OTI-to-Image Retrieval</h2></summary> 

We use **OTI** as a mapping tecnique from visual to textual features.
  
#### OTI: Inversion + Evaluation

To **map** the query visual features of a selected dataset, simply run the following command:
```bash
python src/oti.py \
    --config configs/oti/{CONFIG}.yaml \  # e.g configs/oti/clip_vib32.yaml    
    --dataroot /PATH/TO/DATASETS/ \     
    --dataset_name {DATASET_NAME}         # e.g. oxford_pets
```

```
--config CONFIG   Path to config file to load. Available configs can be found inside '/configs/oti'.
--dataroot DATAROOT   Root directory containing all datasets.
--dataset_name DATASET_NAME   Name of the dataset to evaluate.
```
To automatically run evaluation at the end of the inversion simply include  `--validate True` in the command above.

To modify any default configuration parameter, simply check the editable parameters available in the config file and override them by passing `--PARAM_NAME value` in the command line.

Editable parameters include:

```
--split SPLIT   Dataset split to use (e.g., 'test', 'val'). Defaults to the same split used in the paper.
--exp_name EXP_NAME   Name of the experiment. Used to name output folders and checkpoints.
--clip_model_name CLIP_MODEL_NAME   CLIP model variant to use, e.g., 'ViT-B/32'.
--resume_experiment   Resume training and logging for the same experiment if it already exists.
--seed SEED   Seed value for reproducibility. Use a positive value to ensure deterministic behavior.
--validate   If set, run evaluation on the validation set instead of training.
--learning_rate LEARNING_RATE   Learning rate for optimization (e.g., 0.02).
--batch_size BATCH_SIZE   Batch size used during training and optimization.
--oti_steps OTI_STEPS   Number of optimization steps for generating OTI features.
--num_pseudo_tokens NUM_PSEUDO_TOKENS   Number of pseudo tokens used in OTI (e.g., 1).
--save_frequency SAVE_FREQUENCY   Frequency (in steps) at which model checkpoints are saved.
--weight_decay WEIGHT_DECAY   Weight decay used during optimization to regularize training (e.g., 0.01).
--template_sentence TEMPLATE_SENTENCE   Template used to construct sentences in OTI (e.g., 'a photo of {}').
--use_open_clip   If set, use OpenCLIP instead of the standard OpenAI CLIP implementation.
--open_clip_pretrained OPEN_CLIP_PRETRAINED   Name of the OpenCLIP pretrained weights (e.g., 'laion2b_s34b_b79k').
```

#### OTI: Only Evaluation
If you have already the OTI-inverted features and you only want to evaluate the retrieval performances, simply run the following command:

```bash
python src/retrieval.py \
    --dataroot /PATH/TO/DATASETS/ \
    --dataset_name {DATASET_NAME} \         # e.g. oxford_pets
    --clip_model_name {CLIP_MODEL_NAME} \   # e.g. ViT-B/32
    --query_eval_type oti \
    --gallery_eval_type image \
    --query_exp_name {QUERY_EXP_NAME}
```
```
  --dataroot DATAROOT   Root directory containing all datasets.
  --dataset_name DATASET_NAME   Name of the dataset to evaluate.
  --clip_model_name CLIP_MODEL_NAME   CLIP model variant to use, e.g. 'ViT-B/32'.
  --query_eval_type {oti,ovi,image,text}
                        Type of feature used for query: 'oti' for OTI-inverted features, 'ovi' for OVI-inverted features, 'image' for original image
                        encoder features, 'text' for original text encoder features.
  --gallery_eval_type {oti,ovi,image,text}
                        Type of feature used for gallery: 'oti' for OTI-inverted features, 'ovi' for OVI-inverted features, 'image' for original image
                        encoder features, 'text' for original text encoder features.
  --oti_template_sentence OTI_TEMPLATE_SENTENCE   Template sentence used in OTI for generating textual pseudo-tokens (e.g., 'a photo of {}').
  --query_split QUERY_SPLIT   Dataset split used for query samples (e.g., 'train', 'test').
  --gallery_split GALLERY_SPLIT   Dataset split used for gallery samples (e.g., 'train', 'test').
  --query_exp_name QUERY_EXP_NAME   Experiment name for loading precomputed OTI/OVI query features.
  --gallery_exp_name GALLERY_EXP_NAME   Experiment name for loading precomputed OTI/OVI gallery features.
  --use_open_clip       If set, use OpenCLIP instead of the standard OpenAI CLIP implementation.
  --open_clip_pretrained OPEN_CLIP_PRETRAINED   Name of the pretrained weights for OpenCLIP (e.g., 'laion2b_s34b_b79k').
```

#### Image-to-Image Retrieval Baseline

To reproduce the standard image-to-image retrieval baseline using raw CLIP features, simply run the following command:
```bash
python src/retrieval.py \
    --dataroot /PATH/TO/DATASETS/ \
    --dataset_name {DATASET_NAME} \         # e.g. oxford_pets
    --clip_model_name {CLIP_MODEL_NAME} \   # e.g. ViT-B/32
    --query_eval_type image \
    --gallery_eval_type image \
    --query_exp_name {QUERY_EXP_NAME}
```
</details>

<details>
<summary><h2>OVI-to-Text Retrieval</h2></summary> 

We use **OVI** as a mapping tecnique from textual to visual features.

#### OVI: Inversion + Evaluation

To **invert** the query text features of a selected dataset, simply run the following command:
```bash
python src/ovi.py \
    --config configs/ovi/{CONFIG}.yaml \    # configs/ovi/clip_vib32.yaml
    --dataroot /PATH/TO/DATASETS/ \
    --dataset_name {DATASET_NAME} \         # e.g. nocaps_text
```
Analogously to OTI, to automatically run evaluation at the end of the inversion simply include  `--validate True` in the command above.

Analogously to OTI, to modify any default configuration parameter, simply check the editable parameters available in the config file and override them by passing `--PARAM_NAME value` in the command line (see previous OTI section).

Please note that classification datasets do not contain text features to invert. As written in the paper for the classification datasets, we use the classnames in the format `"A photo of [CLS_NAME]."`
To run OVI on classification datasets include `--use_classnames True` in the command line.

#### OVI: Only Evaluation
If you have already inverted the text features and you only want to evaluate the retrieval performances of such OVI features, simply run the following command:
```bash
python src/retrieval.py \
    --dataroot /PATH/TO/DATASETS/ \
    --dataset_name {DATASET_NAME} \         # e.g. nocaps_text
    --clip_model_name {CLIP_MODEL_NAME} \   # e.g. ViT-B/32
    --query_eval_type ovi \
    --gallery_eval_type text \
    --query_exp_name {QUERY_EXP_NAME}
```
In the previous OTI section we listed all the available editable configuration parameters.

#### Text-to-Text Retrieval Baseline

To reproduce the standard text-to-text retrieval baseline using raw CLIP features, simply run the following command:
```bash
python src/retrieval.py \
    --dataroot /PATH/TO/DATASETS/ \
    --dataset_name {DATASET_NAME} \         # e.g. nocaps_text
    --clip_model_name {CLIP_MODEL_NAME} \   # e.g. ViT-B/32
    --query_eval_type text \
    --gallery_eval_type text \
    --query_exp_name {QUERY_EXP_NAME}
```

</details>

<details open>
<summary><h2>OTI Minimal Working Example</h2></summary>

```python
import torch
# remember to use our load_clip function to load the model
from utils import load_clip
from oti import oti
from PIL import Image

# check if cuda is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# todo change with your image path
image_path = ""

# choose the model you want to use
clip_model_name = "ViT-B/32"

# remember to use our load_clip function to load the model
clip_model, _, preprocess = load_clip(clip_model_name)

# preprocess the image with the clip model preprocessing function
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

# extract image features
image_features = clip_model.encode_image(image).float()

# define your template sentence here
template_sentence = "A photo of {}."

# use oti to invert the image features into textual tokens
# inversion can take some time, so be patient
oti_pseudo_tokens, loss = oti(image_features.to(device), clip_model, template_sentence=template_sentence)

# extract the OTI-inverted features
# note that "£" will be replaced with the oti-pseudo tokens
texts = clip_model.tokenizer(template_sentence.format(" £ ")).to(device)

# extract the OVI-inverted features
oti_features = clip_model.encode_with_pseudo_tokens(texts, oti_pseudo_tokens)

# calculate the cosine similarity between the original image features and the OTI-inverted features
cosine_similarity = torch.cosine_similarity(image_features, oti_features)
print(f"Cosine similarity between image features and OTI-inverted feature: {cosine_similarity.item()}")
```
</details>


<details open>
<summary><h2>OVI Minimal Working Example</h2></summary>

```python
import torch
# remember to use our load_clip function to load the model
from utils import load_clip
from ovi import ovi
from PIL import Image

# check if cuda is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# todo change with your text to invert
text = "This is a sample text to invert!"

clip_model_name = "ViT-B/32"

# remember to use our load_clip function to load the model
clip_model, _, preprocess = load_clip(clip_model_name)

# extract the text features
text_features = clip_model.encode_text(clip_model.tokenizer(text).to(device)).float()

# use ovi to invert the text features into visual tokens
# inversion can take some time, so be patient
ovi_pseudo_tokens, loss = ovi(text_features.to(device), clip_model)

# extract the OVI-inverted features
ovi_features = clip_model.encode_image_with_pseudo_tokens(ovi_pseudo_tokens).float()

# calculate the cosine similarity between the original text features and the OVI-inverted features
cosine_similarity = torch.cosine_similarity(text_features, ovi_features)
print(f"Cosine similarity between text features and OVI-inverted feature: {cosine_similarity.item()}")
```
</details>

## Authors
* [**Marco Mistretta**](https://scholar.google.com/citations?hl=it&user=KMIb4eAAAAAJ)**\***
* [**Alberto Baldrati**](https://scholar.google.com/citations?hl=en&user=I1jaZecAAAAJ)**\***
* [**Lorenzo Agnolucci**](https://scholar.google.it/citations?user=hsCt4ZAAAAAJ&hl)**\***
* [**Marco Bertini**](https://scholar.google.it/citations?user=SBm9ZpYAAAAJ&hl=it)
* [**Andrew David Bagdanov**](https://scholar.google.com/citations?user=_Fk4YUcAAAAJ&hl=en)

**\*** Equal contribution.
