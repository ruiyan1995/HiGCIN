import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import utils
import argparse
from torch.autograd import Variable
from .lib.non_local_dot_product import NONLocalBlock2D
import Models

__all__ = ['HiGCIN', 'hiGCIN']

parser = argparse.ArgumentParser()
parser.add_argument('--body_infer', type=utils.str2bool, default=True, choices=[True, False], help='do inference or not')
parser.add_argument('--body_infer_mode', type=str, default='Cross', choices=['All', 'Cross'])
parser.add_argument('--person_infer', type=utils.str2bool, default=True, choices=[True, False], help='do inference or not')
parser.add_argument('--person_infer_mode', type=str, default='Cross', choices=['All', 'Cross'])
parser.add_argument('--backbone_name', type=str, default='resNet18', choices=['alexNet', 'resNet18'])
opt, _ = parser.parse_known_args()

class hiGCIN(nn.Module):
    def __init__(self, dataset_name, model_confs, mode):
        super(hiGCIN, self).__init__()
        self.mode = mode
        self.backbone_name = opt.backbone_name
        if 'alexNet' in self.backbone_name or 'vgg' in self.backbone_name:
            self.input_size = 4096
        elif 'resNet18' in self.backbone_name:
            self.input_size = 512
        elif 'inception' in self.backbone_name:
            self.input_size = 2048
        self.num_players = model_confs.num_players
        self.num_actions = model_confs.num_actions
        self.num_classes = model_confs.num_classes
        self.num_frames = model_confs.num_frames
        self.num_groups = model_confs.num_groups
        self.body_infer = opt.body_infer
        self.body_infer_mode = opt.body_infer_mode
        self.person_infer = opt.person_infer
        self.person_infer_mode = opt.person_infer_mode
        self.phase = None

        if self.mode == 'end_to_end':
            # extract feature from CNN model directly
            self.backbone = eval('Models.'+self.backbone_name)(True, dataset_name, model_confs = model_confs, body_infer = self.body_infer, body_infer_mode = self.body_infer_mode, do_LSTM = False, do_Classify = False)
            
        if self.person_infer:
            self.PIM = NONLocalBlock2D(in_channels=self.input_size, sub_sample=False, bn_layer=True, model_confs=model_confs, mode=self.person_infer_mode, S=self.num_players)
            self.dropout = nn.Dropout()
            

        self.fea_size = self.input_size
        self.output_size = self.fea_size*self.num_groups
        w = self.num_players//self.num_groups
        self.pool = nn.MaxPool2d((w, 1), stride = (w, 1))
        self.action_classifier = nn.Linear(self.fea_size, self.num_actions)
        self.activity_classifier = nn.Sequential(
            nn.Linear(self.output_size, self.num_classes) # + self.num_players*4
        )

    def forward(self, inputs):
        # inputs: (b, T, c, h, w)
        self.phase = 'trainval' if self.training else 'test'
        self.T = self.num_frames[self.phase]
        b = inputs.size()[0]
        img = inputs.view(-1, *inputs.size()[2:])
        if self.mode == 'end_to_end':
            x, BIM_f = self.backbone(img)

        x = x.view(x.size(0), self.num_players, x.size(1)//self.num_players) # bt,k,d

        if self.person_infer:
            x = x.view(x.size(0)//self.T, self.T, *x.size()[-2:]) # b,t,k,d
            x = x.permute(0, 3, 1, 2) # x = (b, d, t, k)
            x, PIM_f = self.PIM(x) # x=(b, d, t, k)
            x = self.dropout(x)
            x = x.permute(0, 2, 3, 1).contiguous() # x = (b, t, k, d)
            x = x.view(x.size(0)*self.T, *x.size()[-2:]) # x=(b*t, k, d)
        else:
            PIM_f = None

        
        action_scores = self.action_classifier(x.view(-1, x.size(-1)))
        
        if self.num_groups==2:
            lx, rx = torch.chunk(x, 2, 1) # lx (N, K/2, d)
            # do intra-group pooling
            x = torch.cat((lx.unsqueeze(1),rx.unsqueeze(1)), 3)
            
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        x = torch.squeeze(x)
        activity_scores = self.activity_classifier(x)
        return x, action_scores, activity_scores, BIM_f, PIM_f


def HiGCIN(pretrained=False, dataset_name=None, **kwargs):
    #pretrained=True
    model = hiGCIN(dataset_name, **kwargs)

    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = torch.load('./weights/'+dataset_name+'/activity/ResNet18/Ours_R_BIM_PIM/best_wts.pkl')
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model