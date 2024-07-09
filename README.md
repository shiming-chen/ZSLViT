# ZSLViT 


This repository contains the training and testing code for the CVPR'24 paper titled with  "***Progressive Semantic-Guided Vision Transformer for  Zero-Shot Learningg***".


## Requirements
The code implementation of **ZSLViT** mainly based on [PyTorch](https://pytorch.org/). All of our experiments run and test in Python 3.9.7. To install all required dependencies:
```
$ pip install -r requirements.txt
```


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


## Train
Runing following commands and training **ZSLViT**:

Need to modify the wandb_config file.

```
$ python train.py
```

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


## Acknowledgement :heart:
This project is based on EViT ([paper](https://arxiv.org/abs/2202.07800)) and ViT-ZSL([paper](https://arxiv.org/abs/2108.00045)). Thanks for their wonderful works.

## Citation
If you find ZSLViT is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

```bibtex
@inproceedings{chen2024progressive,
  title={Progressive Semantic-Guided Vision Transformer for Zero-Shot Learning},
  author={Chen, Shiming and Hou, Wenjin and Khan, Salman and Khan, Fahad Shahbaz},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={23964--23974},
  year={2024}
}
```
