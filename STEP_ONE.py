"""
	Group Activity Recognition
"""

import argparse
import Runtime
import torch
import time


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default='./dataset/', help='Please set the root folder of datasets')
parser.add_argument('--dataset_name', type=str, default='VD', choices=['VD', 'CAD'], help='Please choose one of the dataset')
opt, _ = parser.parse_known_args()

# Inference to activity directly, end to end
Activity = Runtime.Activity_Level(opt.dataset_root, opt.dataset_name, 'end_to_end')
Activity.trainval()
