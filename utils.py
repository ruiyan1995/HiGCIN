# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt
import itertools
from sklearn import datasets, svm, metrics
from collections import deque
import pickle
from PIL import Image, ImageDraw
from operator import truediv

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def get_avg_acc(preds, targets):
    confusion = metrics.confusion_matrix(targets, preds)
    each_acc = np.nan_to_num(truediv(np.diag(confusion), np.sum(confusion, axis=1)))
    return each_acc, np.mean(each_acc)

def normlize(matrix):
    matrix = np.asarray(matrix, dtype=float)
    return np.round(matrix/np.sum(matrix, 1).reshape(-1,1)*100, decimals=2)

def save_pkl(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_pkl(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_mask(img, candidate, scale=5):
    mask = np.ones((img.size[1], img.size[0]))
    for joint_coords in candidate:
        #print joint_coords
        #(x, y) = min(tuple(joint_coords[0:2].astype(int)-scale),0)
        #(x_, y_) = tuple(joint_coords[0:2].astype(int)+scale)
        x = max(joint_coords[0]-scale, 0)
        y = max(joint_coords[1]-scale, 0)
        x_ = min(joint_coords[0]+scale, img.size[0])
        y_ = min(joint_coords[1]+scale, img.size[1])
        mask[y:y_,x:x_]=0
    return mask

def get_partial_body(img, mask):
    person_img = img.copy()
    sub_img = Image.fromarray(np.uint8(255*np.zeros((person_img.size[1], person_img.size[0], 3))))
    mask = Image.fromarray(np.uint8(255*(mask)))
    person_img.paste(sub_img, (0, 0), mask)
    return person_img

def get_pos_feas(positions, d_emb):
    """
    Args:
        positions: [N,2]
    Returns:
        positions high-dimensional representation: [N,d_emb]
    """

    N=positions.shape[0]
    
    d=d_emb//2
    
    idxs = [np.power(1000,2*(idx//2)/d) for idx in range(d)]
    idxs = torch.FloatTensor(idxs).to(device=positions.device)
    
    idxs = idxs.repeat(N,2)  #N, d_emb
    
    pos = torch.cat([ positions[:,0].reshape(-1,1).repeat(1,d),positions[:,1].reshape(-1,1).repeat(1,d) ],dim=1)

    embeddings=pos/idxs
    
    embeddings[:,0::2]=torch.sin(embeddings[:,0::2])  # dim 2i
    embeddings[:,1::2]=torch.cos(embeddings[:,1::2])  # dim 2i+1
    
    return embeddings

def write_txt(txtFile, content_str, mode):
    with open(txtFile, mode) as f:
        f.write(content_str)

def annotation_parse(line, action_list=[], activity_list=[]):
    keywords = deque(line.strip().split(' '))
    frame_id = keywords.popleft().split('.')[0]
    activity = activity_list.index(keywords.popleft())
    Rects = []
    while keywords:
        x = int(keywords.popleft())
        y = int(keywords.popleft())
        w = int(keywords.popleft())
        h = int(keywords.popleft())
        action = action_list.index(keywords.popleft())
        Rects.append([x,y,w,h,action,activity])
    Rects = np.asarray(Rects)
    # sort Rects by the first col
    Rects = Rects[np.lexsort(Rects[:,::-1].T)]
    return frame_id, Rects


def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=100):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def block_shuffle(List, block_length):
    """
        Args
    """
    # print(len(List), block_length)
    base_indx = np.arange(0, len(List), block_length)
    np.random.shuffle(base_indx)
    indx = base_indx
    for i in range(block_length - 1):
        new_indx = base_indx + i + 1
        indx = np.column_stack((indx, new_indx))
    indx = indx.reshape(-1)
    # print indx
    # print List.type()
    #shuffled_List = List[indx]
    shuffled_List = type(List)(map(lambda i: List[i], indx))
    return shuffled_List



def get_confusion_matrix(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    matrix = matrix * 100 / matrix.astype(np.float).sum(axis=1).reshape(-1, 1)
    return matrix



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='',
                          cmap=plt.cm.Greys):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    fmt = '.2f' if normalize else '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    #plt.ylabel('True label')
    #plt.xlabel('Predicted label')


def Min_Max_Normlize(vec):
    """
        Args: vec
    """
    Max, Min = torch.max(vec), torch.min(vec)
    eps = 0.0001
    return (vec - Min) / (Max - Min)

