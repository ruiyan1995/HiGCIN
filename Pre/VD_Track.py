#coding=utf-8
import os
import glob
import numpy as np
import cv2
import sys
sys.path.append('..')
import utils
from .Track import *
import time


class VD_Track(Track):
    """docstring for VD_Preprocess"""
    def __init__(self, dataset_root, dataset_confs, model_confs=None, interval=None):
        super(VD_Track, self).__init__(dataset_root, dataset_confs, 'VD', model_confs)
        self.num_videos = dataset_confs.num_videos
        if interval:
            # track the persons
            start, end = interval
            self.getPersons(start, end)
        else:
            # write the train_test file
            self.getTrainTest()



    def getPersons(self, start, end):
        # Create the correlation tracker - the object needs to be initialized
        # before it can be used
        
        for video_id in range(start, end):
            video_id = str(video_id)
            annotation_file = os.path.join(self.dataset_folder, video_id, 'annotations.txt')
            f = open(annotation_file)
            lines = f.readlines()
            imgs={}
            for line in lines:
                frame_id, rects = utils.annotation_parse(line, self.action_list, self.activity_list)
                img_list = sorted(glob.glob(os.path.join(self.dataset_folder, video_id, frame_id, "*.jpg")))[16:26]
                imgs['pre'] = img_list[:5][::-1]
                imgs['back'] = img_list[4:]
                since = time.time()
                if len(rects)<=self.K_players:
                    print (video_id, frame_id)
                    save_path = os.path.join(self.save_folder, video_id, frame_id)
                    if not(os.path.exists(save_path)):
                        os.makedirs(save_path)
                        # We will track the frames as we load them off of disk
                        self.track(rects, imgs, self.tracker, save_path)
                print(time.time()-since)
#             exit(0)
        

    def getTrainTest(self):

        # split train-test following [CVPR 16]
        dataset_config={'trainval':self.trainval_videos,'test':self.test_videos}
        dataList={'trainval':[],'test':[]}
        activity_label_list = {'trainval':[],'test':[]}

        print ('trainval_videos:', dataset_config['trainval'])
        print ('test_videos:', dataset_config['test'])

        for phase in ['trainval','test']:
            action_list = []
            activity_list = []
            for idx in dataset_config[phase]:
                video_path = os.path.join(self.save_folder, str(idx))
                print(video_path)
                for root, dirs, files in os.walk(video_path):
                    activity_label = ''
                    if len(files)!=0:
                        files.sort()
                        for i in range(self.K_players*self.T):
                            if i<len(files):
                                # parse
                                filename = files[i]
                                action_label = filename.split('_')[1]
                                activity_label = filename.split('_')[2].split('.')[0]
                                content_str = root + '/' + filename + '\t' + action_label + '\n'
                                action_list.append(content_str)
                                content_str = root + '/' + filename + '\t' + activity_label + '\n'
                                activity_list.append(content_str)
                            else:
                                # add none.jpg
                                filename = self.dataset_root + 'none.jpg'
                                content_str = filename + '\t' + 'error' + '\n'
                                action_list.append(content_str)
                                content_str = filename + '\t' + activity_label + '\n'
                                activity_list.append(content_str)

            # print(action_list, self.T, '_action', phase)
            self.write_list(action_list, self.T, '_action', phase)
            self.write_list(activity_list, self.T*self.K_players, '_activity', phase)