import numpy as np
import matplotlib.pyplot as plt                        
import torch
import torchvision.models as models
import torch.nn as nn
import os
import scipy.io as sio
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from Dataset import DataLoader,trainTransform,testTransform
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from test_function import *
from zslvit import ZSLViT,_cfg
import torch.backends.cudnn as cudnn
from timm.models import create_model
import utils
import wandb

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

wandb.init(project='VIT-ZSL',config='wandb_config/awa2_zslvit.yaml')
args = wandb.config
print('Config file from wandb:', args)

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
test_seen_data_loader = torch.utils.data.DataLoader(test_seen_data, batch_size=256, shuffle=False, num_workers=num_workers)
# unseen test data loader
test_unseen_data = DataLoader(ROOT, test_unseen_files, test_unseen_labels, transform=testTransform)
test_unseen_data_loader = torch.utils.data.DataLoader(test_unseen_data, batch_size=256, shuffle=False, num_workers=num_workers)

utils.init_distributed_mode(args)
seed = args.seed + utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
cudnn.benchmark = True
zslvit = create_model(
   args.model,
   base_keep_rate=args.base_keep_rate,
   drop_loc=eval(args.drop_loc),
   pretrained=False,
   num_classes=args.nb_classes,
   drop_rate=args.drop,
   drop_path_rate=args.drop_path,
   drop_block_rate=None,
   fuse_token=args.fuse_token,
   img_size=(args.input_size, args.input_size),
   dataset = args.dataset
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
    "vit": zslvit,
    "mlp_g": mlp_g})

if args.gzsl:
   checkpoint = torch.load('/home/admin/chenshiming/hwj/ViT-ZSL/saved_model/ZSLViT_AWA2_GZSL.pth',map_location=torch.device('cpu'))
else:
   checkpoint = torch.load('/home/admin/chenshiming/hwj/ZSLVIT/saved_model/ZSLViT_AWA2_CZSL.pth',map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
model.eval()

if use_cuda:
    model = model.cuda()


if DATASET == 'AWA2':
  gamma = args.gamma
elif DATASET == 'CUB':
  gamma = args.gamma
elif DATASET == 'SUN':
  gamma = args.gamma
else:
  print("Please specify the dataset, and set {attr_length} equal to the attribute length")

test(args,model, test_seen_data_loader, test_seen_labels, test_unseen_data_loader, test_unseen_labels, attrs_mat, use_cuda, gamma)

   
