# Dataset Preparation Guide

This guide provides instructions for setting up all datasets required to run experiments in Cross-the-Gap.

## Overview

Cross-the-Gap supports **27 datasets** across multiple task categories:

### Task Categories and Datasets

**Image-to-Image Retrieval (15 datasets):**
- CUB-200-2011
- Stanford Online Products
- ROxford5k and RParis6k (with Revisitop1m 1M distractors)
- **11 Classification Datasets** (also used for retrieval): Caltech-101, DTD, EuroSAT, FGVC Aircraft, Food-101, ImageNet, Oxford Flowers-102, Oxford Pets, Stanford Cars, SUN397, UCF-101

**Text-to-Text Retrieval (12 datasets):**
- COCO (MS-COCO 2014)
- Flickr30K
- NoCaps
- IMDB Reviews
- 20 Newsgroups
- **7 NanoBEIR datasets**: ClimateFEVER, DBPedia, FEVER, NFCorpus, NQ, SCIDOCS, SciFact

**Zero-Shot Classification (11 datasets):**
- Caltech-101, DTD, EuroSAT, FGVC Aircraft, Food-101, ImageNet, Oxford Flowers-102, Oxford Pets, Stanford Cars, SUN397, UCF-101

**Note**: The 11 classification datasets serve dual purposes—they are used for both image-to-image retrieval experiments (test vs. train split) and zero-shot classification tasks.

We recommend organizing all datasets under a single root directory (e.g., `$DATA`) for convenience. If datasets are stored elsewhere, symbolic links can be created:

```bash
# If datasets are at /path/to/datasets/
cd Cross-the-Gap/
ln -s /path/to/datasets/ $DATA
```

---

## Table of Contents

1. [Classification Datasets (CoOp Format)](#classification-datasets-coop-format)
2. [CUB-200-2011](#cub-200-2011)
3. [Stanford Online Products](#stanford-online-products)
4. [ROxford5k and RParis6k](#roxford5k-and-rparis6k)
5. [Flickr30K](#flickr30k)
6. [COCO (MS-COCO 2014)](#coco-ms-coco-2014)
7. [NoCaps](#nocaps)
8. [IMDB Reviews](#imdb-reviews)
9. [20 Newsgroups](#20-newsgroups)
10. [NanoBEIR](#nanobeir)

---

## Classification Datasets (CoOp Format)

For the 11 classification datasets used in zero-shot evaluation, we follow the [CoOp Dataset Preparation Guide](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md).

### Supported Classification Datasets

- ImageNet
- Caltech-101
- Oxford Pets
- Stanford Cars
- Oxford Flowers-102
- Food-101
- FGVC Aircraft
- SUN397
- Descriptive Textures Dataset (DTD)
- EuroSAT
- UCF-101

### Quick Setup

Please follow the comprehensive instructions at:
**https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md**

The CoOp guide provides:
- Download links for images and annotations
- Train/val/test split files for reproducibility
- Correct directory structure for each dataset

**Important**: Ensure all classification datasets are placed under `$DATA/` with the exact folder names specified in the CoOp guide.

---

## CUB-200-2011

CUB-200-2011 (Caltech-UCSD Birds-200-2011) is a fine-grained bird species classification and retrieval dataset containing 200 bird categories.

- **Images**: 11,788 bird images (5,994 train + 5,794 test)
- **Task**: Image-to-image retrieval
- **Official Website**: [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/)

### Installation

1. **Create the dataset directory**:
```bash
mkdir -p $DATA/CUB_200_2011
cd $DATA/CUB_200_2011
```

2. **Download the dataset**:
```bash
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
tar -xzf CUB_200_2011.tgz
mv CUB_200_2011/* .
rm -r CUB_200_2011
rm CUB_200_2011.tgz
```

### Expected Directory Structure

```
CUB_200_2011/
├── images/
│   ├── 001.Black_footed_Albatross/
│   │   ├── Black_Footed_Albatross_0001_796111.jpg
│   │   ├── ...
│   ├── 002.Laysan_Albatross/
│   ├── ...
│   └── 200.Common_Yellowthroat/
├── images.txt
├── image_class_labels.txt
├── train_test_split.txt
├── classes.txt
├── bounding_boxes.txt
└── ...
```

### Required Files

- `images.txt`: Maps image ID to filepath
- `image_class_labels.txt`: Maps image ID to class label
- `train_test_split.txt`: Specifies train/test split (1=train, 0=test)

---

## Stanford Online Products

Stanford Online Products (SOP) is a large-scale dataset for metric learning and product retrieval, containing 120,053 product images across 22,634 classes organized into 12 super-classes.

- **Images**: 120,053 product images (59,551 train + 60,502 test)
- **Classes**: 11,318 train classes + 11,316 test classes
- **Super-classes**: 12 categories (bicycle, cabinet, chair, coffee maker, fan, kettle, lamp, mug, sofa, stapler, table, toaster)
- **Task**: Image-to-image retrieval with metric learning
- **Official Paper**: [Deep Metric Learning via Lifted Structured Feature Embedding](https://cvgl.stanford.edu/projects/lifted_struct/)

### Installation

1. **Create the dataset directory**:
```bash
mkdir -p $DATA/Stanford_Online_Products
cd $DATA/Stanford_Online_Products
```

2. **Download the dataset**:
```bash
# Download from the official FTP server
wget ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip
unzip Stanford_Online_Products.zip
mv Stanford_Online_Products/* .
rm -r Stanford_Online_Products
rm Stanford_Online_Products.zip
```

**Alternative**: Download manually from [Stanford Online Products Dataset](https://cvgl.stanford.edu/projects/lifted_struct/) if FTP is unavailable.

### Expected Directory Structure

```
Stanford_Online_Products/
├── bicycle_final/
│   ├── 111085122871_0.JPG
│   ├── ...
├── cabinet_final/
├── chair_final/
├── coffee_maker_final/
├── fan_final/
├── kettle_final/
├── lamp_final/
├── mug_final/
├── sofa_final/
├── stapler_final/
├── table_final/
├── toaster_final/
├── Ebay_train.txt
└── Ebay_test.txt
```

### Required Files

- `Ebay_train.txt`: Training split with format: `image_id class_id super_class_id path`
- `Ebay_test.txt`: Test split with the same format

### Format of Split Files

Each line in `Ebay_train.txt` and `Ebay_test.txt` follows this format:
```
image_id class_id super_class_id path
0 1 1 bicycle_final/111085122871_0.JPG
1 1 1 bicycle_final/111085122871_1.JPG
...
```

- `image_id`: Sequential image identifier
- `class_id`: Fine-grained product class (22,634 total)
- `super_class_id`: Super-class category (1-12)
- `path`: Relative path to image

---

## ROxford5k and RParis6k

ROxford5k and RParis6k are revisited versions of the Oxford5k and Paris6k landmark retrieval datasets, featuring improved annotations and standardized evaluation protocols.

- **ROxford5k**: 4,993 database images + 70 query images
- **RParis6k**: 6,322 database images + 70 query images
- **Revisitop1m**: 1,001,001 distractor images
- **Task**: Large-scale instance-level image retrieval
- **Official Paper**: [Revisiting Oxford and Paris: Large-Scale Image Retrieval Benchmarking](https://arxiv.org/abs/1803.11285)
- **Official Repository**: [filipradenovic/revisitop](https://github.com/filipradenovic/revisitop)

### Installation

**Option 1: Using Official Python Script (Recommended)**

The official repository provides automated download scripts:

```bash
# Set your data directory path
export DATA=/path/to/your/datasets  # Change this to your actual path
# Example: export DATA=$HOME/datasets or export DATA=/mnt/data/datasets

# Clone the official repository
git clone https://github.com/filipradenovic/revisitop.git
cd revisitop/python

# Download ROxford5k and RParis6k
python3 -c "from download import download_datasets; download_datasets('${DATA}')"

# Download Revisitop1m (1M distractors)
python3 -c "from download import download_distractors; download_distractors('${DATA}')"
```

This will create datasets in a `datasets/` subdirectory. Move them to the main data directory:

```bash
# Move datasets out of the datasets/ subdirectory
cd ${DATA}
mv datasets/roxford5k .
mv datasets/rparis6k .
mv datasets/revisitop1m .

# Remove the empty datasets/ directory
rmdir datasets
```

**Final structure:**
```
$DATA/
├── roxford5k/
│   ├── jpg/
│   └── gnd_roxford5k.pkl
├── rparis6k/
│   ├── jpg/
│   └── gnd_rparis6k.pkl
└── revisitop1m/
    └── jpg/
```

**Option 2: Manual Download**

#### 1. ROxford5k

```bash
mkdir -p $DATA/roxford5k
cd $DATA/roxford5k
```

**Download images**:
```bash
# Download Oxford5k images
wget https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz
tar -xzf oxbuild_images.tgz
# Images extract directly to current directory, move them to jpg/
mkdir -p jpg
mv *.jpg jpg/
rm oxbuild_images.tgz
```

**Download ground truth annotations**:
```bash
# Download ROxford5k ground truth
wget http://cmp.felk.cvut.cz/revisitop/data/datasets/roxford5k/gnd_roxford5k.pkl
```

#### 2. RParis6k

```bash
mkdir -p $DATA/rparis6k
cd $DATA/rparis6k
```

**Download images**:
```bash
# Download Paris6k images
wget https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_1.tgz
wget https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_2.tgz
tar -xzf paris_1.tgz
tar -xzf paris_2.tgz
# Create jpg directory and move all images
mkdir -p jpg
mv paris/*/*.jpg jpg/
rm -r paris paris_1.tgz paris_2.tgz
```

**Download ground truth annotations**:
```bash
# Download RParis6k ground truth
wget http://cmp.felk.cvut.cz/revisitop/data/datasets/rparis6k/gnd_rparis6k.pkl
```

#### 3. Revisitop1m (1M Distractors)

The Revisitop1m dataset adds 1 million distractor images to increase retrieval difficulty.

```bash
mkdir -p $DATA/revisitop1m
cd $DATA/revisitop1m
```

**Download image list**:
```bash
wget http://ptak.felk.cvut.cz/revisitop/revisitop1m/jpg/revisitop1m.txt
```

**Download images**:
```bash
# Create jpg directory
mkdir -p jpg

# Download all 100 parts (this will take time)
# Note: Files are numbered 1-100, not 0-99
for i in {1..100}; do
    wget http://ptak.felk.cvut.cz/revisitop/revisitop1m/jpg/revisitop1m.${i}.tar.gz
done

# Extract all archives
for i in {1..100}; do
    tar -xzf revisitop1m.${i}.tar.gz -C jpg/
    rm revisitop1m.${i}.tar.gz  # Remove after extraction to save space
done
```

**Alternative**: Download in batches to manage storage, or use the official Python script (see Option 1 above).

### Expected Directory Structure

```
roxford5k/
├── jpg/
│   ├── all_souls_000000.jpg
│   ├── all_souls_000001.jpg
│   ├── ...
└── gnd_roxford5k.pkl

rparis6k/
├── jpg/
│   ├── paris_defense_000000.jpg
│   ├── paris_defense_000001.jpg
│   ├── ...
└── gnd_rparis6k.pkl

revisitop1m/
├── jpg/
│   ├── 0/
│   │   ├── 0a0c58e3c0ce3fb1.jpg
│   │   ├── ...
│   ├── 1/
│   ├── ...
│   └── f/
└── revisitop1m.txt
```

### Ground Truth Format

The `.pkl` files contain dictionaries with:
- `imlist`: List of database image names (without extension)
- `qimlist`: List of query image names (without extension)
- `gnd`: List of ground truth annotations per query with keys:
  - `ok`: Positive matches (medium difficulty)
  - `junk`: Junk images to ignore
  - `easy`: Easy positive matches (optional)
  - `hard`: Hard positive matches (optional)

---

## Flickr30K

Flickr30K is a widely-used image captioning dataset containing 31,783 images with 5 human-annotated captions per image.

- **Images**: 31,783 images (158,915 captions total)
- **Captions**: 5 captions per image
- **Splits**: Uses Karpathy splits (29,000 train, 1,014 val, 1,000 test)
- **Task**: Text-to-text retrieval (caption-to-caption)
- **Official Paper**: [From image descriptions to visual denotations](https://www.aclweb.org/anthology/Q14-1006/)
- **Usage in Cross-the-Gap**:
  - Uses **validation split** for retrieval experiments (`val_query` vs `val_gallery`)
  - Only captions are used (images not required)

### Installation

**Note**: For text-to-text retrieval tasks in Cross-the-Gap, **only the caption files are required**. Images are optional and not used by the dataloader.

1. **Create the dataset directory**:
```bash
mkdir -p $DATA/Flickr30K
cd $DATA/Flickr30K
```

2. **Download Karpathy splits** (Required):
```bash
# Download train/val/test splits
wget https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
unzip caption_datasets.zip

# Create split files from Karpathy annotations
python3 << 'EOF'
import json

with open('dataset_flickr30k.json', 'r') as f:
    data = json.load(f)

splits = {'train': [], 'val': [], 'test': []}
for item in data['images']:
    # Image filename without extension (required format)
    img_name = item['filename'].replace('.jpg', '')
    splits[item['split']].append(img_name)

# Save splits as text files (REQUIRED by dataloader)
for split_name, img_list in splits.items():
    with open(f'karpathy_{split_name}.txt', 'w') as f:
        f.write('\n'.join(img_list))

print(f"Split files created:")
print(f"  karpathy_train.txt: {len(splits['train'])} images")
print(f"  karpathy_val.txt: {len(splits['val'])} images")
print(f"  karpathy_test.txt: {len(splits['test'])} images")
EOF

# Clean up
rm caption_datasets.zip
```

3. **Generate captions file** (Required):

**Option A: Download from Kaggle** (requires authentication):
```bash
# Visit: https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset
# Download "flickr30k_images.zip" which contains results.csv
unzip flickr30k_images.zip
# Keep results.csv and discard other files
rm -r flickr30k_images/  # Remove duplicate images if present
rm flickr30k_images.zip
```

**Option B: Generate from Karpathy annotations** (recommended if Kaggle access unavailable):
```bash
# Generate results.csv from dataset_flickr30k.json
python3 << 'EOF'
import json
import csv

# Load the dataset
with open('dataset_flickr30k.json', 'r') as f:
    data = json.load(f)

# Create results.csv
with open('results.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter='|')
    csvwriter.writerow(['image_name', ' comment_number', ' comment'])

    for image in data['images']:
        filename = image['filename']
        for idx, sentence in enumerate(image['sentences']):
            caption = sentence['raw']
            csvwriter.writerow([filename, idx, f' {caption}'])

print(f"Generated results.csv with {len(data['images']) * 5} captions")
EOF

# Optional: remove dataset_flickr30k.json after generating results.csv
# rm dataset_flickr30k.json
```

4. **Download images** (Optional - only if needed for other tasks):
```bash
# Images are NOT required for text-to-text retrieval in Cross-the-Gap
# If you need them for other purposes:
# Visit: https://shannon.cs.illinois.edu/DenotationGraph/
# Fill form to request access and download flickr30k-images.tar.gz
tar -xzf flickr30k-images.tar.gz
mv flickr30k-images images  # Optional: rename for consistency
```

### Expected Directory Structure

**Minimum required for text-to-text retrieval:**
```
Flickr30K/
├── results.csv                # Required: 5 captions per image
├── karpathy_train.txt         # Required: train split image names
├── karpathy_val.txt           # Required: val split image names
├── karpathy_test.txt          # Required: test split image names
└── dataset_flickr30k.json     # Optional: Karpathy annotations (for reference)
```

**Full structure (if images are also downloaded):**
```
Flickr30K/
├── images/                    # Optional: not used by text retrieval
│   ├── 1000092795.jpg
│   ├── 10002456.jpg
│   ├── ...
│   └── 997722733.jpg
├── results.csv                # Required
├── karpathy_train.txt         # Required
├── karpathy_val.txt           # Required
├── karpathy_test.txt          # Required
└── dataset_flickr30k.json     # Optional
```

### Format of results.csv

The `results.csv` file uses pipe (`|`) delimiter with 5 captions per image:
```
image_name | comment_number | comment
1000092795.jpg|0| Two young guys with shaggy hair look at their hands while hanging out in the yard .
1000092795.jpg|1| Two young , White males are outside near many bushes .
1000092795.jpg|2| Two young people wearing glasses and white shirts stand together.
1000092795.jpg|3| A man in a blue shirt is standing next to someone.
1000092795.jpg|4| Two friends enjoy time spent together.
...
```

**Important Notes**:
- Each image has exactly 5 captions (comment_number 0-4)
- For text-to-text retrieval:
  - **Query split** uses the first caption (comment_number 0)
  - **Gallery split** uses captions 2-5 (comment_number 1-4)
- Image `2199200615.jpg` may have a malformed delimiter in some CSV versions (handled automatically by the dataloader)

---

## COCO (MS-COCO 2014)

Microsoft COCO (Common Objects in Context) is a large-scale dataset for object detection, segmentation, and captioning.

- **Images**: 82,783 train images + 40,504 val images
- **Captions**: 5 captions per image (~616,000 captions)
- **Task**: Text-to-text retrieval (caption-to-caption)
- **Official Website**: [COCO Dataset](https://cocodataset.org/)

### Installation

1. **Create the dataset directory**:
```bash
mkdir -p $DATA/COCO2014
cd $DATA/COCO2014
```

2. **Download Karpathy split annotations** (required for text retrieval):

The Karpathy split is widely used for image captioning and provides train/val/test splits with caption annotations.

```bash
# Download Karpathy splits
wget http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
unzip caption_datasets.zip

# Extract COCO annotations
# Both files are useful - dataset_coco.json contains full annotations
cp dataset_coco.json dataset_coco_light.json

# Clean up
rm caption_datasets.zip
```

**Note**: For text-to-text retrieval in Cross-the-Gap, only `dataset_coco_light.json` is required (images not needed).

3. **Download images** (optional, only if you need images for other tasks):
```bash
# Download training images
wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip
rm train2014.zip

# Download validation images
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
rm val2014.zip
```

### Expected Directory Structure

```
COCO2014/
├── dataset_coco.json          # Full Karpathy annotations
├── dataset_coco_light.json    # Required for text retrieval
├── train2014/                 # (optional, only for image tasks)
│   ├── COCO_train2014_000000000009.jpg
│   ├── COCO_train2014_000000000025.jpg
│   └── ...
└── val2014/                   # (optional, only for image tasks)
    ├── COCO_val2014_000000000042.jpg
    ├── COCO_val2014_000000000073.jpg
    └── ...
```

### Format of dataset_coco_light.json

The `dataset_coco_light.json` file contains:
```json
{
  "images": [
    {
      "imgid": 0,
      "split": "train",
      "sentences": [
        {"sentid": 0, "raw": "A caption here..."},
        ...
      ]
    },
    ...
  ]
}
```

- `split`: Can be `train`, `val`, `test`, or `restval`
- Our dataloader merges `train` and `restval` for the training split

---

## NoCaps

NoCaps (Novel Object Captioning at Scale) is a challenging image captioning dataset designed to test generalization to novel objects.

- **Images**: 15,100 validation images from Open Images
- **Captions**: 10 human-annotated captions per image (4,500 used in validation)
- **Domains**: In-domain, Near-domain, Out-of-domain splits
- **Task**: Text-to-text retrieval (caption-to-caption)
- **Official Website**: [NoCaps](https://nocaps.org/)

### Installation

1. **Create the dataset directory**:
```bash
mkdir -p $DATA/nocaps
cd $DATA/nocaps
```

2. **Download annotations** (required for text retrieval):

Visit the [NoCaps download page](https://nocaps.org/download) and download:
- Validation annotations (`nocaps_val_4500_captions.json`)

```bash
# Download from Google Drive (official NoCaps download page)
curl -L "https://drive.google.com/uc?export=download&id=1zrsLhT94nazJ5h-LleHsU4AojHBqpcA0" -o nocaps_val_4500_captions.json

# Alternative: Visit https://nocaps.org/download to get the latest download link
```

3. **Download validation images** (optional, only needed for image-related tasks):

```bash
# Download validation images
wget https://nocaps.s3.amazonaws.com/nocaps_val_4500_images.zip
unzip nocaps_val_4500_images.zip -d val
rm nocaps_val_4500_images.zip
```

**Note**: For text-to-text retrieval tasks in Cross-the-Gap, only the captions JSON file is required. NoCaps only provides validation and test splits publicly. The test set annotations are held privately for evaluation via the official server.

### Expected Directory Structure

```
nocaps/
├── nocaps_val_4500_captions.json   # Required for text retrieval
├── nocaps_test_image_info.json     # (optional)
└── val/                             # (optional, only for image tasks)
    ├── 0a000a6e-e4cd-4070-bd21-9b8b4d4b37e6.jpg
    ├── 0a0088f6-c77f-42be-b677-7dc6056a0e15.jpg
    └── ...
```

### Format of nocaps_val_4500_captions.json

```json
{
  "annotations": [
    {
      "id": 0,
      "image_id": "000001",
      "caption": "A caption here..."
    },
    ...
  ]
}
```

---

## IMDB Reviews

IMDB Movie Reviews dataset for sentiment analysis and text retrieval.

- **Reviews**: 50,000 movie reviews (25,000 train + 25,000 test)
- **Task**: Text-to-text retrieval with sentiment
- **Source**: [Stanford AI Lab](http://ai.stanford.edu/~amaas/data/sentiment/)

### Installation

The IMDB dataset will be downloaded automatically when running experiments. However, you can also download it manually:

1. **Create the dataset directory**:
```bash
mkdir -p $DATA/IMDB
cd $DATA/IMDB
```

2. **Download the dataset**:
```bash
# Download from Kaggle
# Visit: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
# Download "IMDB Dataset.csv"
```

Or use the official source:
```bash
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzf aclImdb_v1.tar.gz
```

3. **Convert to CSV format** (if using official source):

```python
import pandas as pd
import os

def load_imdb_data(data_dir):
    reviews, labels = [], []
    for label in ['pos', 'neg']:
        folder = os.path.join(data_dir, label)
        for filename in os.listdir(folder):
            with open(os.path.join(folder, filename), 'r', encoding='utf-8') as f:
                reviews.append(f.read())
                labels.append('positive' if label == 'pos' else 'negative')
    return reviews, labels

train_reviews, train_labels = load_imdb_data('aclImdb/train')
test_reviews, test_labels = load_imdb_data('aclImdb/test')

df = pd.DataFrame({
    'review': train_reviews + test_reviews,
    'sentiment': train_labels + test_labels
})
df.to_csv('IMDB_Dataset.csv', index=False)
```

### Expected Directory Structure

```
IMDB/
└── IMDB_Dataset.csv
```

---

## 20 Newsgroups

The 20 Newsgroups dataset is a collection of newsgroup documents for text classification.

- **Documents**: ~20,000 documents
- **Categories**: 20 newsgroup topics
- **Task**: Text-to-text retrieval across topics
- **Source**: Automatically downloaded via scikit-learn

### Installation

No manual installation required! The dataset is automatically downloaded via `sklearn.datasets.fetch_20newsgroups()` when running experiments.

The dataset will be cached in your scikit-learn data directory (typically `~/scikit_learn_data/`).

---

## NanoBEIR

NanoBEIR is a collection of 7 text retrieval benchmarks, scaled down for efficient evaluation.

- **Datasets**: ClimateFEVER, DBPedia, FEVER, NFCorpus, NQ, SCIDOCS, SciFact
- **Task**: Text-to-text information retrieval
- **Source**: HuggingFace datasets

### Installation

No manual installation required! The NanoBEIR datasets are automatically downloaded from HuggingFace when running experiments.

The datasets will be cached in your HuggingFace cache directory (typically `~/.cache/huggingface/datasets/`).

### Supported NanoBEIR Datasets

1. **NanoClimateFEVER**: Climate change claims verification
2. **NanoDBPedia**: Entity retrieval from DBPedia
3. **NanoFEVER**: Fact verification
4. **NanoNFCorpus**: Medical information retrieval
5. **NanoNQ**: Natural Questions retrieval
6. **NanoSCIDOCS**: Scientific document retrieval
7. **NanoSciFact**: Scientific fact verification

---

## Citation

If you use these datasets, please cite the original papers.

## Questions?

If you encounter any issues with dataset setup, please open an issue on our [GitHub repository](https://github.com/miccunifi/Cross-the-Gap/issues).
