"""
    Data_Configs
"""
import argparse
import os


class Data_Configs(object):
    """docstring for Dataset_Configs"""

    def __init__(self, dataset_root, dataset_name, stage, mode):
        super(Data_Configs, self).__init__()
        self.dataset_root = dataset_root
        self.dataset_name = dataset_name
        self.stage = stage
        self.mode = mode

        self.confs_dict = {
            'action': {
                # you can set it as the case may be, such as 'img', 'npy',
                # 'hdf5', and so on.
                'data_type': {
                    'trainval_action': 'img',
                    'extract_action_feas': 'img',
                    'frame_trainval_activity': 'img'
                },
                'cur_folder': {
                    'trainval_action': 'imgs',
                    'extract_action_feas': 'imgs',
                    'frame_trainval_activity': 'imgs'
                },
                'label_type': {'trainval_action': 'action', 'extract_action_feas': 'activity', 'frame_trainval_activity': 'frame_activity'},
                'batch_size': {'trainval_action': {'trainval': 160, 'test': 10}, 'extract_action_feas': {'trainval': 120, 'test': 120} }
            },
            'activity': {
                'data_type': {
                    'trainval_activity': 'npy',
                    'end_to_end': 'img'
                },
                'cur_folder': {
                    'trainval_activity': 'feas',
                    'end_to_end': 'imgs'
                },
                'label_type': {'trainval_activity': 'activity', 'end_to_end':'activity'},
                #'batch_size': {'trainval_activity': {'trainval': 500, 'test': 10}, 'end_to_end': {'trainval': 120*4, 'test': 120}} # for VD(Alex)
                'batch_size': {'trainval_activity': {'trainval': 500, 'test': 10}, 'end_to_end': {'trainval': 4, 'test': 1}} # for VD(Resnet-18)

                #'batch_size': {'trainval_activity': {'trainval': 500, 'test': 10}, 'end_to_end': {'trainval': 120*4, 'test': 120}} # for VD(Resnet-18) NLNN
                #'batch_size': {'trainval_activity': {'trainval': 500, 'test': 10}, 'end_to_end': {'trainval': 120*2, 'test': 120}} # for VD(Inception)
                #'batch_size': {'trainval_activity': {'trainval': 500, 'test': 10}, 'end_to_end': {'trainval': 120*4, 'test': 120}} # for VD(VGG16)
#                 'batch_size': {'trainval_activity': {'trainval': 500, 'test': 10}, 'end_to_end': {'trainval': 60*8, 'test': 60}} #CAD
            }
        }

    def configuring(self):
        dataset_confs = self.confs_dict[self.stage]
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset_folder', type=str, default=os.path.join(
            self.dataset_root, self.dataset_name, dataset_confs['cur_folder'][self.mode]), help='')
        parser.add_argument('--batch_size', type=dict,
                            default=dataset_confs['batch_size'][self.mode])

        parser.add_argument('--data_type', type=str, default=dataset_confs[
                            'data_type'][self.mode], choices=['img', 'hdf5', 'npy'], help='the story type for data')
        parser.add_argument('--label_type', type=str, default=dataset_confs['label_type'][self.mode], help='the label types for data')
        opt, _ = parser.parse_known_args()
        return opt
