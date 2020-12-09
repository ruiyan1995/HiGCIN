"""
	Dataset_preprocessing
"""
import Configs
from .VD_Track import *
from .CAD_Track import *

class Processing(object):
    """Preprocessing dataset, e.g., track, split and anonatation."""

    def __init__(self, dataset_root, dataset_name, interval):
        super(Processing, self).__init__()
        self.dataset_root = dataset_root
        self.dataset_name = dataset_name

        # Pre configs:
        dataset_confs = Configs.Dataset_Configs(dataset_root, dataset_name).configuring()
        model_confs = Configs.Model_Configs(dataset_name, 'action').configuring()
        eval(self.dataset_name + '_Track')(self.dataset_root, dataset_confs, model_confs, interval)
