"""
	Solver_Configs
"""
import argparse
import torch
import torch.nn as nn
from torch.optim import lr_scheduler


class Solver_Configs(object):
    """docstring for Solver_Configs"""

    def __init__(self, dataset_name, data_loaders, data_sizes, net, stage, mode, dataset_confs):
        super(Solver_Configs, self).__init__()
        self.dataset_name = dataset_name
        self.dataset_confs = dataset_confs
        
        self.data_loaders = data_loaders
        self.data_sizes = data_sizes
        self.stage = stage
        self.net = net
        self.mode = mode

        self.confs_dict = {
            'VD': {
                'action': {
                    'criterion': {'trainval_action': 'nn.CrossEntropyLoss', 'extract_action_feas': 'nn.CrossEntropyLoss'},
                    'num_epochs': {
                        'trainval_action': 20,
                        'extract_action_feas': 20
                    },
                    'lr_scheduler': {
                        'trainval_action': {'step_size': 5, 'gamma': 0.1},
                        'extract_action_feas': {'step_size': 5, 'gamma': 0.1}
                    },
                    'optimizer': {
                        'trainval_action': {'method': 'SGD', 'lr': 0.001, 'arg': 0.9},
                        'extract_action_feas': {'method': 'SGD', 'lr': 0.001, 'arg': 0.9}
                    }
                },
                'activity': {
                    'criterion': {'trainval_activity': 'nn.CrossEntropyLoss', 'end_to_end': 'nn.CrossEntropyLoss'},
                    'num_epochs': {
                        'trainval_activity': 50,
                        'end_to_end': 15
                    },
                    'lr_scheduler': {
                        'trainval_activity': {'step_size': 10, 'gamma': 0.1},
                        'end_to_end': {'step_size': 5, 'gamma': 0.1}
                    },
                    'optimizer': {
                        'trainval_activity': {'method': 'Adam', 'lr': 0.0001, 'arg': (0.9, 0.9)},
#                         'end_to_end': {'method': 'SGD', 'lr': 0.0001, 'arg': 0.9}, # init learning rate = 0.0001
                        'end_to_end': {'method': 'Adam', 'lr': 0.0001, 'arg': (0.9, 0.9)}, # init learning rate = 0.0001
                    }
                }
            },

            'CAD': {
                'action': {
                    'criterion': {'trainval_action': 'nn.CrossEntropyLoss', 'extract_action_feas': 'nn.CrossEntropyLoss'},
                    'num_epochs': {
                        'trainval_action': 50,
                        'extract_action_feas': 0
                    },
                    'lr_scheduler': {
                        'trainval_action': {'step_size': 7, 'gamma': 0.5},
                        'extract_action_feas': {'step_size': 5, 'gamma': 0.1}
                    },
                    'optimizer': {
                        #'trainval_action': {'method': 'SGD', 'lr': 0.001, 'arg': 0.9},
                        'trainval_action': {'method': 'Adam', 'lr': 0.0001, 'arg': (0.9, 0.9)},
                        'extract_action_feas': {'method': 'SGD', 'lr': 0.001, 'arg': 0.9}
                    }
                },
                'activity': {
                    'criterion': {'trainval_activity': 'nn.CrossEntropyLoss', 'end_to_end': 'nn.CrossEntropyLoss'},
                    'num_epochs': {
                        'trainval_activity': 50,
                        'end_to_end': 5
                    },
                    'lr_scheduler': {
                        'trainval_activity': {'step_size': 10, 'gamma': 0.1},
                        'end_to_end': {'step_size': 5, 'gamma': 0.1}
                    },
                    'optimizer': {
                        'trainval_activity': {'method': 'Adam', 'lr': 0.0001, 'arg': (0.9, 0.9)},
                        'end_to_end': {'method': 'SGD', 'lr': 0.0001, 'arg': 0.9},
                        #'end_to_end': {'method': 'Adam', 'lr': 0.0001, 'arg': (0.9, 0.9)}
                    }
                }
            }
        }

    def configuring(self):
        solver_confs = self.confs_dict[self.dataset_name][self.stage]
        parser = argparse.ArgumentParser()
        parser.add_argument('--num_epochs', type=int, default=solver_confs['num_epochs'][self.mode])
        parser.add_argument('--gpu', type=bool, default=torch.cuda.is_available(), help='*****')

        criterion = eval(solver_confs['criterion'][self.mode])()
        parser.add_argument('--criterion', type=type(criterion), default=criterion)

        optim = solver_confs['optimizer'][self.mode]
        optimizer = eval('torch.optim.' + optim['method'])(self.net.parameters(), optim['lr'], optim['arg'], weight_decay=1e-4) # weight_decay=1e-4
        parser.add_argument('--optimizer', type=type(optimizer), default=optimizer)

        lr_sch = solver_confs['lr_scheduler'][self.mode]
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_sch['step_size'], gamma=lr_sch['gamma'])
        parser.add_argument('--exp_lr_scheduler', type=type(exp_lr_scheduler), default=exp_lr_scheduler)
        parser.add_argument('--dataset_name', type=str, default=self.dataset_name)
        parser.add_argument('--data_loaders', type=type(self.data_loaders), default=self.data_loaders)
        parser.add_argument('--data_sizes', type=type(self.data_sizes), default=self.data_sizes)
        parser.add_argument('--stage', type=str, default=self.stage)
        parser.add_argument('--mode', type=str, default=self.mode)
        parser.add_argument('--label_type', type=type(self.dataset_confs.label_type), default=self.dataset_confs.label_type)
        

        opt, _ = parser.parse_known_args()
        return opt
