"""
"""
import torch
from torch.utils.data import Dataset
from skimage import io, transform
import numpy as np
import os
from PIL import Image
import utils
import random


class VD_img(Dataset):
    def __init__(self, data_confs, model_confs, phase, data_transform=None):
        self.dataset_folder = data_confs.dataset_folder
        self.label_type = data_confs.label_type
        self.num_frames = model_confs.num_frames
#         self.used_frames = model_confs.used_frames
        self.num_players = model_confs.num_players
        self.phase = phase
        self.data_transform = data_transform
        lines = open(os.path.join(self.dataset_folder, phase + '_' + self.label_type + '.txt'))

        self.path_list = []
        self.labels_list = []


        for i, line in enumerate(lines):
            img_path = line.split('\n')[0].split('\t')[0]
            img_path = os.path.join('/'.join(self.dataset_folder.split('/')[:-2]), '/'.join(img_path.split('/')[2:]))
            activity_label = line.split('\n')[0].split('\t')[1]
            
            #########################################################
            if 'none' not in img_path.split('/')[-1]:
                action_label = img_path.split('/')[-1].split('_')[1]
            else:
                action_label = 9
            #########################################################
                
            if 'error' in activity_label:
                pass
            else:
                self.path_list.append(img_path)
                self.labels_list.append([int(action_label), int(activity_label)])

        
#     def __getitem__(self, index):
#         file_name = self.path_list[index]
#         img = Image.open(file_name)
#         img = self.data_transform(img) if self.data_transform else img
#         action_target, activity_target = self.labels_list[index]
#         return img, [int(action_target), int(activity_target)]

    def __len__(self):
        return len(self.labels_list)//(self.num_frames['test']*self.num_players)
    
    def __getitem__(self, index):
#         interval = self.num_frames*self.num_players
#         img_paths = self.path_list[index*interval:(index+1)*interval]
#         targets = torch.tensor(self.labels_list[index*interval:(index+1)*interval])
        img_paths, targets = self.random_sample(index, sample_number=self.num_frames[self.phase])
        imgs = self.load_sequence(img_paths)

        return imgs, targets, img_paths
    
    def load_sequence(self, img_paths):
        img_list = []
        for img_path in img_paths:
            img = Image.open(img_path)
            img = self.data_transform(img) if self.data_transform else img
            img_list.append(img)
        return torch.stack(img_list)

    def random_sample(self, index, sample_number):
        interval = self.num_frames['test']*self.num_players
        img_paths = self.path_list[index*interval:(index+1)*interval]
        targets = self.labels_list[index*interval:(index+1)*interval]
        random_idxs = self.get_batch_random_idxs(self.num_frames['test'],sample_number,self.num_players)
        sub_img_paths = []
        sub_targets = []
        for idx in random_idxs:
            sub_img_paths.append(img_paths[idx])
            sub_targets.append(targets[idx])
#         print(sub_img_paths)
#         sub_img_paths = []
#         step = len(img_paths)//sample_number
#         for seg_id in range(sample_number):
#             seg_paths = img_paths[seg_id*step:(seg_id+1)*step]
#             selected_path = random.choice(seg_paths) if self.phase=='trainval' else seg_paths[0]
#             sub_img_paths.append(selected_path)
#         #sub_img_paths = random.sample(img_paths, sample_number) if sample_number<self.num_frames else img_paths

        return sub_img_paths, torch.tensor(sub_targets)
    
    def get_batch_random_idxs(self, num_frames=10, sample_number=3, num_players=12):
        a=list(range(num_frames*num_players))
        out = []
        step = num_frames//sample_number
        for seg_id in range(sample_number):
            start = random.randint(seg_id*step,(seg_id+1)*step-1)
            out+=a[start::num_frames]
        out.sort()
        return out