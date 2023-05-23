from tqdm import tqdm
import torch
import numpy as np
from scipy.special import softmax
import torch.nn.functional as F
from helpers import adjust_keep_rate

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def get_reprs(model, data_loader, use_cuda,dataset):
    model.eval()
    reprs = []
    for _, (data, _) in enumerate(data_loader):
        if use_cuda:
            data = data.cuda()
        if dataset == 'SUN':
            with torch.no_grad():
                # only take the global feature
                feat,_,_,_= model.vit(data,keep_rate=(1,1,1,0.9,1,1,0.9,1,1,1,1,1))
        else:
            with torch.no_grad():
                # only take the global feature
                feat,_,_,_= model.vit(data,keep_rate=(1,1,1,0.9,1,1,0.9,1,1,0.9,1,1))
        # global feature

        feat = model.mlp_g(feat)
        reprs.append(feat.cpu().data.numpy())
    reprs = np.concatenate(reprs, 0)
    return reprs

def compute_accuracy(pred_labels, true_labels, labels):
    acc_per_class = np.zeros(labels.shape[0])
    for i in range(labels.shape[0]):
        idx = (true_labels == labels[i])
        acc_per_class[i] = np.sum(pred_labels[idx] == true_labels[idx]) / np.sum(idx)
    return np.mean(acc_per_class)


def test(args,model, test_seen_loader, test_seen_labels, test_unseen_loader, test_unseen_labels, attrs_mat, use_cuda, gamma):
    # Representation
    with torch.no_grad():
        seen_reprs = get_reprs(model, test_seen_loader, use_cuda,dataset=args.dataset)
        unseen_reprs = get_reprs(model, test_unseen_loader, use_cuda,dataset=args.dataset)
    # Labels
    uniq_test_seen_labels = np.unique(test_seen_labels)
    uniq_test_unseen_labels = np.unique(test_unseen_labels)

    # GZSL
    if args.gzsl:
        # Calibrated stacking
        Cs_mat = np.zeros(attrs_mat.shape[0])
        Cs_mat[uniq_test_seen_labels] = gamma
    # seen classes
        gzsl_seen_sim = softmax(seen_reprs @ attrs_mat.T, axis=1) - Cs_mat
        gzsl_seen_predict_labels = np.argmax(gzsl_seen_sim, axis=1)
        gzsl_seen_acc = compute_accuracy(gzsl_seen_predict_labels, test_seen_labels, uniq_test_seen_labels)
        
        # unseen classes
        gzsl_unseen_sim = softmax(unseen_reprs @ attrs_mat.T, axis=1) - Cs_mat
        gzsl_unseen_predict_labels = np.argmax(gzsl_unseen_sim, axis=1)
        gzsl_unseen_acc = compute_accuracy(gzsl_unseen_predict_labels, test_unseen_labels, uniq_test_unseen_labels)

        H = 2 * gzsl_unseen_acc * gzsl_seen_acc / (gzsl_unseen_acc + gzsl_seen_acc)

        
        print('GZSL Seen=%.2f Unseen=%.2f H=%.2f' %(gzsl_seen_acc * 100,gzsl_unseen_acc * 100,H * 100))

    # ZSL
    else:
        zsl_unseen_sim = unseen_reprs @ attrs_mat[uniq_test_unseen_labels].T
        predict_labels = np.argmax(zsl_unseen_sim, axis=1)
        zsl_unseen_predict_labels = uniq_test_unseen_labels[predict_labels]
        zsl_unseen_acc = compute_accuracy(zsl_unseen_predict_labels, test_unseen_labels, uniq_test_unseen_labels)
        print('CZSL Acc=%.2f' %(zsl_unseen_acc * 100))

    

    
    