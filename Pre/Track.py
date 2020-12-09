#coding=utf-8
import os
import glob
import dlib
from collections import deque
from skimage import io, transform
import numpy as np
import cv2
import utils

class Track(object):
    """docstring for Track"""
    def __init__(self, dataset_root, dataset_confs, dataset_name, model_confs=None):
        super(Track, self).__init__()
        self.dataset_root = dataset_root
        self.dataset_folder = os.path.join(dataset_root, dataset_name, 'videos')
        self.K_players = dataset_confs.num_players
        self.T = model_confs.num_frames['test']
        self.action_list = dataset_confs.action_list
        self.activity_list = dataset_confs.activity_list
        self.trainval_videos = dataset_confs.splits['trainval']
        self.test_videos = dataset_confs.splits['test']
        #self.posture = Posture()
        self.tracker = dlib.correlation_tracker()
        #self.win = dlib.image_window()
        self.save_folder = os.path.join(dataset_root, dataset_name, 'imgs')
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
#         self.joints_dict = {}


    def track(self, person_rects, imgs, tracker, save_path):
#         candidate = {}
        for i, person_rect in enumerate(person_rects):
            for j, phase in enumerate(['pre', 'back']):
                if j == 0:
                    j = -1
                for k, f in enumerate(imgs[phase]):
                    #print("Processing Frame {}".format(k))
                    #frame_img = io.imread(f)
                    #print f
                    frame_img = cv2.imread(f)
                    frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
#                     if f not in candidate:
#                         candidate[f] = self.posture.get_candidate_list(frame_img)
                    #posture.plot(candidate[f], frame_img)
                    if k == 0:
                        x, y, w, h, label, group_label = person_rect
                        #print x,y,w,h
                        tracker.start_track(frame_img, dlib.rectangle(int(x),int(y),int(x+w),int(y+h)))
                    else:
                        tracker.update(frame_img)
                    # save imgs
                    #cropped_image = frame_img[Y:Y + HEIGHT,X + biasRate*WIDTH:X + (1-biasRate)*WIDTH,:]
                    pos = tracker.get_position()
                    top, bottom, left, right = max(int(pos.top()),0),max(int(pos.bottom()),0),max(int(pos.left()),0),max(int(pos.right()),0)
                    cropped_image = frame_img[top:bottom,left:right]
#                     new_person_rect = top, bottom, left, right

                    #cropped_image = transform.resize(np.ascontiguousarray(cropped_image),(256,256),mode='constant')
#                     joints = self.posture.get_person_joints(new_person_rect, candidate[f])
                    #posture.plot(joints, cropped_image)
                    img_name = os.path.join(save_path, "%04d_%d_%d.jpg"%(10*i+(5+j*k), label, group_label))
                    #print img_name
                    io.imsave(img_name, cropped_image)
#                     key = str('/'.join(img_name.split('/')[-3:]))
#                     self.joints_dict[key] = joints
                    
    def write_list(self, source_list, block_size, label_type, phase):
        # print(block_size)
        source_list = utils.block_shuffle(source_list, block_size)
        txtFile = os.path.join(self.save_folder, phase + label_type + '.txt')
        open(txtFile, 'w')
        print (phase +'_size:' + str(len(source_list)/(block_size)))
        for i in range(len(source_list)):
            with open(txtFile, 'a') as f:
                f.write(source_list[i])