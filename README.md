# ZSLViT 


This repository contains the testing code for the CVPR'24 paper titled with  "***Progressive Semantic-Guided Vision Transformer for  Zero-Shot Learningg***".



## Preparing Dataset and Model

We provide trained models ([Google Drive](https://drive.google.com/drive/folders/130_RgZndLkLpoP1yqf7CpWbzaO_26XL0?usp=sharing)) on three different datasets: [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [SUN](http://cs.brown.edu/~gmpatter/sunattributes.html), [AWA2](http://cvml.ist.ac.at/AwA2/) in the CZSL/GZSL setting. You can download model files as well as corresponding datasets, and organize them as follows: 
```
.
├── saved_model
│   ├── ZSLViT_CUB_CZSL.pth
│   ├── ZSLViT_CUB_GZSL.pth
│   ├── ZSLViT_SUN_CZSL.pth
│   ├── ZSLViT_SUN_GZSL.pth
│   ├── ZSLViT_AWA2_CZSL.pth
│   └── ZSLViT_AWA2_GZSL.pth
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

## Train
Runing following commands and Training **ZSLViT**:
```
$ python train.py      # CZSL Setting and GZSL Setting 
```

Need to modify the wandb_config file, gzsl is True or False.

## Test
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


| Dataset | Acc(CZSL) | U(GZSL) | S(GZSL) | H(GZSL) |
| :-----: | :-----: | :-----: | :-----: | :-----: |
| CUB | 78.9 | 69.4 | 78.2 | 73.6 |
| SUN | 68.3 | 45.9 | 48.4 | 47.3 |
| AWA2 | 70.7 | 66.1 | 84.6 | 74.2 |

**Note**: We perform experiments on a single NVIDIA Tesla V100 graphic card with 32GB memory.

