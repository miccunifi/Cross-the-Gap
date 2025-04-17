from .coco_text import CocoTextDataset
from .cub import CUB
from .dassl.caltech101 import Caltech101
from .dassl.dtd import DescribableTextures
from .dassl.eurosat import EuroSAT
from .dassl.fgvc_aircraft import FGVCAircraft
from .dassl.food101 import Food101
from .dassl.imagenet import ImageNet
from .dassl.oxford_flowers import OxfordFlowers
from .dassl.oxford_pets import OxfordPets
from .dassl.stanford_cars import StanfordCars
from .dassl.sun397 import SUN397
from .dassl.ucf101 import UCF101
from .dummy_class_names import DummyClassNamesDataset
from .flickr30k_text import Flickr30KTextDataset
from .imdb_text import IMDBTextDataset
from .nano_beir_text import NanoBEIRDataset
from .newsgroup_text import NewsGroupDataset
from .nocaps_text import NoCapsTextDataset
from .roxford_rparis import ROxfordRParisDataset
from .sop import StanfordOnlineProducts

DASSL_DATASETS = ['stanford_cars', 'oxford_pets', 'oxford_flowers', 'fgvc_aircraft', 'dtd', 'eurosat', 'food101',
                  'sun397', 'caltech101', 'ucf101', 'imagenet']

NANOBEIR_DATASETS = ['nanoclimatefever', 'nanodbpedia', 'nanofever', 'nanonfcorpus', 'nanonq',
                     'nanoscidocs', 'nanoscifact']
