import timm
import numpy as np
import matplotlib.pyplot as plt                        
import torch
import torchvision.models as models
import torch.nn as nn
import os,sys
import scipy.io as sio
import pdb
from time import time
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import random
import torchvision.transforms.functional as TF
from train_function import *
from evit import _cfg
from scipy import spatial
#from scipy.special import softmax
import torch.backends.cudnn as cudnn
from timm.models import create_model
import utils
import argparse
import models
import collections
from torch import optim
from helpers import adjust_keep_rate
import wandb
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

wandb.init(project='VIT-ZSL',config='wandb_config/cub_gzsl_train.yaml')
args = wandb.config
print('Config file from wandb:', args)

class DataLoader(Dataset):
    def __init__(self, root, image_files, labels, transform=None):
        self.root  = root
        self.image_files = image_files
        self.labels = labels 
        self.transform = transform

    def __getitem__(self, idx):
        # read the iterable image
        img_pil = Image.open(os.path.join(self.root, self.image_files[idx])).convert("RGB")
        if self.transform is not None:
            img = self.transform(img_pil)
        # label
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.image_files)

# Training Transformations
trainTransform = transforms.Compose([
                        transforms.Resize((448, 448)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])
# Testing Transformations
testTransform = transforms.Compose([
                        transforms.Resize((448, 448)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])




DATASET = args.dataset

if DATASET == 'AWA2':
  ROOT='./data/AWA2/Animals_with_Attributes2/JPEGImages/'
elif DATASET == 'CUB':
  ROOT='./data/CUB/CUB_200_2011/CUB_200_2011/images/'
elif DATASET == 'SUN':
  ROOT='./data/SUN/images/'
else:
  print("Please specify the dataset")

DATA_DIR = f'./data/{DATASET}'
data = sio.loadmat(f'{DATA_DIR}/res101.mat') 

attrs_mat = sio.loadmat(f'{DATA_DIR}/att_splits.mat')
image_files = data['image_files']

if DATASET == 'AWA2':
  image_files = np.array([im_f[0][0].split('JPEGImages/')[-1] for im_f in image_files])
else:
  image_files = np.array([im_f[0][0].split('images/')[-1] for im_f in image_files])


# labels are indexed from 1 as it was done in Matlab, so 1 subtracted for Python
labels = data['labels'].squeeze().astype(np.int64) - 1
train_idx = attrs_mat['train_loc'].squeeze() - 1
val_idx = attrs_mat['val_loc'].squeeze() - 1
trainval_idx = attrs_mat['trainval_loc'].squeeze() - 1
test_seen_idx = attrs_mat['test_seen_loc'].squeeze() - 1
test_unseen_idx = attrs_mat['test_unseen_loc'].squeeze() - 1

# consider the train_labels and val_labels
train_labels = labels[train_idx]
val_labels = labels[val_idx]

# split train_idx to train_idx (used for training) and val_seen_idx
train_idx, val_seen_idx = train_test_split(train_idx, test_size=0.2, stratify=train_labels)
# split val_idx to val_idx (not useful) and val_unseen_idx
val_unseen_idx = train_test_split(val_idx, test_size=0.2, stratify=val_labels)[1]
# attribute matrix
attrs_mat = attrs_mat["att"].astype(np.float32).T

### used for validation
# train files and labels
train_files = image_files[train_idx]
train_labels = labels[train_idx]
uniq_train_labels, train_labels_based0, counts_train_labels = np.unique(train_labels, return_inverse=True, return_counts=True)
# val seen files and labels
val_seen_files = image_files[val_seen_idx]
val_seen_labels = labels[val_seen_idx]
uniq_val_seen_labels = np.unique(val_seen_labels)
# val unseen files and labels
val_unseen_files = image_files[val_unseen_idx]
val_unseen_labels = labels[val_unseen_idx]
uniq_val_unseen_labels = np.unique(val_unseen_labels)

### used for testing
# trainval files and labels
trainval_files = image_files[trainval_idx]
trainval_labels = labels[trainval_idx]
uniq_trainval_labels, trainval_labels_based0, counts_trainval_labels = np.unique(trainval_labels, return_inverse=True, return_counts=True)
# test seen files and labels
test_seen_files = image_files[test_seen_idx]
test_seen_labels = labels[test_seen_idx]
uniq_test_seen_labels = np.unique(test_seen_labels)
# test unseen files and labels
test_unseen_files = image_files[test_unseen_idx]
test_unseen_labels = labels[test_unseen_idx]
uniq_test_unseen_labels = np.unique(test_unseen_labels)

num_workers = 8
### used in validation
# train data loader
train_data = DataLoader(ROOT, train_files, train_labels_based0, transform=trainTransform)
weights_ = 1. / counts_train_labels
weights = weights_[train_labels_based0]
train_sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=train_labels_based0.shape[0], replacement=True)
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, sampler=train_sampler, num_workers=num_workers)
# seen val data loader
val_seen_data = DataLoader(ROOT, val_seen_files, val_seen_labels, transform=testTransform)
val_seen_data_loader = torch.utils.data.DataLoader(val_seen_data, batch_size=256, shuffle=False, num_workers=num_workers)
# unseen val data loader
val_unseen_data = DataLoader(ROOT, val_unseen_files, val_unseen_labels, transform=testTransform)
val_unseen_data_loader = torch.utils.data.DataLoader(val_unseen_data, batch_size=256, shuffle=False, num_workers=num_workers)


### used in testing
# trainval data loader
trainval_data = DataLoader(ROOT, trainval_files, trainval_labels_based0, transform=trainTransform)
weights_ = 1. / counts_trainval_labels
weights = weights_[trainval_labels_based0]
trainval_sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=trainval_labels_based0.shape[0], replacement=True)
trainval_data_loader = torch.utils.data.DataLoader(trainval_data, batch_size=args.batch_size, sampler=trainval_sampler, num_workers=num_workers)
# seen test data loader
test_seen_data = DataLoader(ROOT, test_seen_files, test_seen_labels, transform=testTransform)
test_seen_data_loader = torch.utils.data.DataLoader(test_seen_data, batch_size=64, shuffle=False, num_workers=num_workers)
# unseen test data loader
test_unseen_data = DataLoader(ROOT, test_unseen_files, test_unseen_labels, transform=testTransform)
test_unseen_data_loader = torch.utils.data.DataLoader(test_unseen_data, batch_size=64, shuffle=False, num_workers=num_workers)


# parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
# args = parser.parse_args()
#args.nb_classes = 1000
utils.init_distributed_mode(args)
print(args)
seed = args.seed + utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
cudnn.benchmark = True
print(f"Creating model: {args.model}")
vit = create_model(
   args.model,
   base_keep_rate=args.base_keep_rate,
   drop_loc=eval(args.drop_loc),
   pretrained=args.pretrained,
   num_classes=args.nb_classes,
   drop_rate=args.drop,
   drop_path_rate=args.drop_path,
   drop_block_rate=None,
   fuse_token=args.fuse_token,
   img_size=(args.input_size, args.input_size)
   )

if DATASET == 'AWA2':
  attr_length = args.attr_length
elif DATASET == 'CUB':
  attr_length = args.attr_length
elif DATASET == 'SUN':
  attr_length = args.attr_length
else:
  print("Please specify the dataset, and set {attr_length} equal to the attribute length")

mlp_g = nn.Linear(args.embed_dim, attr_length, bias=args.bias)
use_cuda = torch.cuda.is_available()

model = nn.ModuleDict({
    "vit": vit,
    "mlp_g": mlp_g})
model.train()

# finetune all the parameters
for param in model.parameters():
    param.requires_grad = True

# move model to GPU if CUDA is available
if use_cuda:
    model = model.cuda()

optimizer = torch.optim.Adam([{"params": model.vit.parameters(), "lr": args.lr_vit, "weight_decay": args.weight_decay_vit},
                              {"params": model.mlp_g.parameters(), "lr": args.lr_mlp_g, "weight_decay": args.weight_decay_mlp_g}])
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30], gamma=0.5)


# train attributes
train_attrbs = attrs_mat[uniq_train_labels]
train_attrbs_tensor = torch.from_numpy(train_attrbs)
# trainval attributes
trainval_attrbs = attrs_mat[uniq_trainval_labels]
trainval_attrbs_tensor = torch.from_numpy(trainval_attrbs)
if use_cuda:
    train_attrbs_tensor = train_attrbs_tensor.cuda()
    trainval_attrbs_tensor = trainval_attrbs_tensor.cuda()

if DATASET == 'AWA2':
  gamma = args.gamma
elif DATASET == 'CUB':
  gamma = args.gamma
elif DATASET == 'SUN':
  gamma = args.gamma
else:
  print("Please specify the dataset, and set {attr_length} equal to the attribute length")
print('Dataset:', DATASET, '\nGamma:',gamma)

best_zsl_epoch = 0
best_gzsl_epoch = 0
best_H = 0
best_gzsl_seen_acc = 0
best_gzsl_unseen_acc = 0
best_zsl_acc = 0
for i in range(args.epoch):
    print('Epoch: ', i)
    loss = train(args,model, trainval_data_loader, trainval_attrbs_tensor, optimizer, use_cuda, i,lamb_1=args.lamb_1)
    lr_scheduler.step()
    zsl_unseen_acc, gzsl_seen_acc, gzsl_unseen_acc,H = test(model, test_seen_data_loader, test_seen_labels, test_unseen_data_loader, test_unseen_labels, attrs_mat, use_cuda, gamma)
    if best_zsl_acc <= zsl_unseen_acc:
      best_zsl_epoch = i
      best_zsl_acc = zsl_unseen_acc
    if best_H< H:
       best_gzsl_epoch = i
       best_H = H
       best_gzsl_seen_acc = gzsl_seen_acc
       best_gzsl_unseen_acc = gzsl_unseen_acc
    if i % 10 == 0:
       print('GZSL: epoch=%d, best_seen=%.2f, best_unseen=%.2f, best_h=%.2f || CZSL: epoch=%d,best_zsl_acc = %.2f' % (best_gzsl_epoch, best_gzsl_seen_acc*100, best_gzsl_unseen_acc*100, best_H*100,best_zsl_epoch,best_zsl_acc*100))
    
    wandb.log({
            'epoch': i,
            'loss': loss,
            'acc_unseen': gzsl_unseen_acc,
            'acc_seen': gzsl_seen_acc,
            'H': H,
            'acc_zs': zsl_unseen_acc,
            'best_acc_unseen': best_gzsl_unseen_acc,
            'best_acc_seen': best_gzsl_seen_acc,
            'best_H': best_H,
            'best_acc_zs': best_zsl_acc,
            'gamma':gamma
        })

print('Dataset:', DATASET, '\nGamma:',gamma)
print('the best GZSL epoch',best_gzsl_epoch) 
print('the best GZSL seen accuracy is %.2f' % (best_gzsl_seen_acc*100))
print('the best GZSL unseen accuracy is %.2f' % (best_gzsl_unseen_acc*100))
print('the best GZSL H is %.2f' % (best_H*100))
print('the best CZSL epoch',best_zsl_epoch)
print('the best ZSL unseen accuracy is %.2f' % (best_zsl_acc*100))
       
       
       
     
