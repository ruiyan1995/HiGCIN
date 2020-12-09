"""
	Common Piplines
"""
from abc import ABCMeta, abstractmethod
from torchvision import transforms
import torch
import Configs
import Data
import Models
import Solver


class Piplines(object):
    """docstring for Piplines"""

    def __init__(self, dataset_root, dataset_name, stage, mode):
        super(Piplines, self).__init__()
        self.dataset_root = dataset_root
        self.dataset_name = dataset_name
        self.stage = stage
        self.mode = mode
        self.configuring()
        
    def configuring(self):
        # Dataset configs:
        self.data_confs = Configs.Data_Configs(
            self.dataset_root, self.dataset_name, self.stage, self.mode).configuring()
        print ('data_confs', self.data_confs)
        
        # Model configs:
        self.model_confs = Configs.Model_Configs(
            self.dataset_name, self.stage).configuring()
        
        self.data_loaders, self.data_sizes = self.loadData()
        self.net = self.loadModel()
        
        if torch.cuda.is_available():
            self.net = self.net.cuda()
#             self.net = torch.nn.DataParallel(self.net).cuda()
        print (self.net)
       

        # Solver configs:
        self.solver_confs = Configs.Solver_Configs(self.dataset_name, self.data_loaders, self.data_sizes, self.net, self.stage, self.mode, self.data_confs).configuring()
        print ('solver_confs', self.solver_confs)

        self.solver = Solver.Solver(self.net, self.model_confs, self.solver_confs)

    def loadModel(self, model_confs):
        raise NotImplementedError

    def defineLoss(self, model_confs):
        raise NotImplementedError

    def trainval(self):
        self.solver.train_model()

    def test(self):
        self.solver.test_model()

    def loadData(self, phases=['trainval', 'test']):
        if self.data_confs.data_type == 'img':
            data_transforms = {
                'trainval': transforms.Compose([
                    transforms.Resize((224, 224)), # 299,299 for inception
#                     transforms.Resize((299, 299)), # 299,299 for inception
                    #transforms.RandomResizedCrop(224),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'test': transforms.Compose([
                    transforms.Resize((224, 224)),
#                     transforms.Resize((299, 299)), # 299,299 for inception
                    #transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            }
        else:
            data_transforms = None

        dataset = {phase: eval('Data.' + self.dataset_name + '_' + self.data_confs.data_type)(
            self.data_confs, self.model_confs, phase, data_transforms[phase] if data_transforms else None) for phase in phases}
        data_loaders = {phase: torch.utils.data.DataLoader(dataset[phase],batch_size=self.data_confs.batch_size[phase], num_workers=8, shuffle=True) for phase in phases}
        # num_workers=8
        data_sizes = {phase: len(dataset[phase]) for phase in phases}
        return data_loaders, data_sizes

    
    def evaluate(self, model_path=None):
        if model_path:
            pretrained_dict = torch.load(model_path)
        else:
            pretrained_dict = torch.load('./weights/'+self.dataset_name+'/xxx.pth')
        self.net.load_state_dict(pretrained_dict)
        self.net.eval()
        self.solver.evaluate()