"""
	Track and crop persons from the videos.
"""

import argparse
import torch
import Pre
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default='./dataset/', help='Please set the root folder of datasets')
parser.add_argument('--dataset_name', type=str, default='VD', choices=['VD', 'CAD'], help='Please choose one of the dataset')
parser.add_argument('--interval', type=int, nargs='+', default=None, help='Please choose one of the dataset')
opt, _ = parser.parse_known_args()

# Step Zero: Dataset Preprocessing
Pre.Processing(opt.dataset_root, opt.dataset_name, interval=opt.interval)




