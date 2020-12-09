from .Piplines import *
class Activity_Level(Piplines):
    """docstring for Action_Level"""
    def __init__(self, dataset_root, dataset_name, mode):
        super(Activity_Level, self).__init__(dataset_root, dataset_name, 'activity', mode)

        
    def extractFeas(self, save_folder):
        pass
        print ('Done, the features files are saved at ' + save_folder + '\n')


    def loadModel(self, pretrained=False):
        net = Models.HiGCIN(pretrained, self.dataset_name, model_confs=self.model_confs, mode=self.mode)
        return net

    def loss(self):
        pass