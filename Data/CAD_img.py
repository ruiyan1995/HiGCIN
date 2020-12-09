"""
 Load data of CAD dataset from img source.
"""
from torch.utils.data import Dataset
from skimage import io, transform
import numpy as np
import os
from PIL import Image
import utils
import random

class CAD_img(Dataset):
    def __init__(self, dataset_folder, phase, label_type, data_transform=None):
        self.scale = 8
        self.label_types = label_type
        self.phase = phase
        self.data_transform = data_transform
        self.joints_dict = utils.load_pkl('./dataset/CAD/pose_imgs/pose')
        lines = open(os.path.join(dataset_folder, phase + '_' + label_type + '.txt'))

        self.path_list = []
        self.labels_list = []
                
        for i, line in enumerate(lines):
            img_path = line.split('\n')[0].split('\t')[0]
            label = line.split('\n')[0].split('\t')[1]

            if 'error' in label:
                pass
            else:
                self.path_list.append(img_path)
                self.labels_list.append(label)

    def __getitem__(self, index):
        r"""
            Make a mask for original Image!
        """
        if self.phase=='trainval':
            self.do_mask = bool(random.getrandbits(1))
            #self.do_mask = False
        else:
            self.do_mask = False
            
        file_name = self.path_list[index]
        person_img = Image.open(file_name)
        if self.do_mask and not ('none' in file_name.split('/')[-1]):
            key = str('/'.join(file_name.split('/')[-3:]))
            joints = self.joints_dict[key]
            # get mask. scale =500 for test,,
            if len(joints)>8:
                mask = utils.get_mask(person_img, joints, scale=self.scale)
                person_img = utils.get_partial_body(person_img, mask)

        if self.data_transform is not None:
            person_img = self.data_transform(person_img)
            
        img = person_img

        target = self.labels_list[index]
        ###################################################
        #pos = self.pos_dict[self.frame_key_list[index]] # [, 2]
        #pos = np.resize(pos, (self.num_players,2))
        ##################################################
        
        return img, int(target)


    def __len__(self):
        return len(self.labels_list)