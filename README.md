# ZSLViT 


This repository contains the testing code for the NeurIPS'23 submission ***ID-3294*** titled with  "***ZSLViT: Semantic-Guided Vision Transformer for  Zero-Shot Learningg***".



## Preparing Dataset and Model

We provide trained models ([Google Drive](https://drive.google.com/drive/folders/1rNHCglaSD_Q5se1rs5qIh6QNtMDCZokc?usp=sharing)) on three different datasets: [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [SUN](http://cs.brown.edu/~gmpatter/sunattributes.html), [AWA2](http://cvml.ist.ac.at/AwA2/) in the CZSL/GZSL setting. You can download model files as well as corresponding datasets, and organize them as follows: 
```
.
├── saved_model
│   ├── TransZeroPP_CUB_CZSL.pth
│   ├── TransZeroPP_CUB_GZSL.pth
│   ├── TransZeroPP_SUN_CZSL.pth
│   ├── TransZeroPP_SUN_GZSL.pth
│   ├── TransZeroPP_AWA2_CZSL.pth
│   └── TransZeroPP_AWA2_GZSL.pth
├── data
│   ├── CUB/
│   ├── SUN/
│   └── AWA2/
└── ···
```

## Requirements
The code implementation of **TransZero++** mainly based on [PyTorch](https://pytorch.org/). All of our experiments run and test in Python 3.8.8. To install all required dependencies:
```
$ pip install -r requirements.txt
```
## Runing
Runing following commands and testing **TransZero++** on different dataset:

CUB Dataset: 
```
$ python test.py --config config/CUB_CZSL.json      # CZSL Setting
$ python test.py --config config/CUB_GZSL.json      # GZSL Setting
```
SUN Dataset:
```
$ python test.py --config config/SUN_CZSL.json      # CZSL Setting
$ python test.py --config config/SUN_GZSL.json      # GZSL Setting
```
AWA2 Dataset: 
```
$ python test.py --config config/AWA2_CZSL.json     # CZSL Setting
$ python test.py --config config/AWA2_GZSL.json     # GZSL Setting
```

## Results
Results of our released models using various evaluation protocols on three datasets, both in the conventional ZSL (CZSL) and generalized ZSL (GZSL) settings.

| Dataset | Acc(CZSL) | U(GZSL) | S(GZSL) | H(GZSL) |
| :-----: | :-----: | :-----: | :-----: | :-----: |
| CUB | 79.5 | 69.8 | 78.2 | 73.8 |
| SUN | 67.8 | 53.0 | 42.2 | 47.0 |
| AWA2 | 67.9 | 62.4 | 87.9 | 73.0 |

**Note**: We perform experiments on a single NVIDIA A100 graphic card with 40GB memory.

