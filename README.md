# ZSLViT 


This repository contains the testing code for the NeurIPS'23 submission ***ID-3294*** titled with  "***ZSLViT: Semantic-Guided Vision Transformer for  Zero-Shot Learningg***".



## Preparing Dataset and Model

We provide trained models ([Google Drive](https://drive.google.com/drive/folders/1rNHCglaSD_Q5se1rs5qIh6QNtMDCZokc?usp=sharing)) on three different datasets: [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [SUN](http://cs.brown.edu/~gmpatter/sunattributes.html), [AWA2](http://cvml.ist.ac.at/AwA2/) in the CZSL/GZSL setting. You can download model files as well as corresponding datasets, and organize them as follows: 
```
.
├── saved_model
│   ├── ZSLViT_CUB_CZSL_224.pth
│   ├── ZSLViT_CUB_GZSL_224.pth
│   ├── ZSLViT_SUN_CZSL_224.pth
│   ├── ZSLViT_SUN_GZSL_224.pth
│   ├── ZSLViT_AWA2_CZSL_224.pth
│   └── ZSLViT_AWA2_GZSL_224.pth
├── data
│   ├── CUB/
│   ├── SUN/
│   └── AWA2/
└── ···
```

## Requirements
The code implementation of **ZSLViT** mainly based on [PyTorch](https://pytorch.org/). All of our experiments run and test in Python 3.9.7. To install all required dependencies:
```
$ pip install -r requirements.txt
```
## Runing
Runing following commands and testing **ZSLViT** on different dataset:

Need to modify the wandb_config file, gzsl is True or False.


CUB Dataset: 
```
$ python test_CUB.py      # CZSL Setting and GZSL Setting 
```
SUN Dataset:
```
$ python test_SUN.py      # CZSL Setting and GZSL Setting 
```
AWA2 Dataset: 
```
$ python test_AWA2.py     # CZSL Setting and GZSL Setting 
```

## Results
Results of our released models using various evaluation protocols on three datasets, both in the conventional ZSL (CZSL) and generalized ZSL (GZSL) settings.

Input size: 224 * 224

| Dataset | Acc(CZSL) | U(GZSL) | S(GZSL) | H(GZSL) |
| :-----: | :-----: | :-----: | :-----: | :-----: |
| CUB | 74.3 | 64.0 | 74.0 | 68.7 |
| SUN | 67.2 | 49.2 | 41.9 | 45.3 |
| AWA2 | 69.2 | 63.8 | 84.5 | 72.7 |

Input size: 448 * 448

| Dataset | Acc(CZSL) | U(GZSL) | S(GZSL) | H(GZSL) |
| :-----: | :-----: | :-----: | :-----: | :-----: |
| CUB | 79.5 | 69.8 | 78.2 | 73.8 |
| SUN | 68.4 | 53.0 | 42.2 | 47.0 |
| AWA2 | 71.1 | 65.2 | 83.5 | 73.2 |

**Note**: We perform experiments on a single NVIDIA Tesla V100 graphic card with 32GB memory.

