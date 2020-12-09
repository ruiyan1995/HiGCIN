import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
from Models.activity.lib.non_local_dot_product import NONLocalBlock2D
import torch.nn.functional as F
__all__ = ['AlexNet', 'alexNet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, model_confs, body_infer=True, body_infer_mode='Cross', do_LSTM=False, do_Classify=False):
        super(AlexNet, self).__init__()
        self.do_LSTM = do_LSTM
        self.do_Classify = do_Classify
        self.num_frames = model_confs.num_frames
        self.K = model_confs.num_players
        self.body_infer = body_infer
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        if self.body_infer:
            self.BIM = NONLocalBlock2D(in_channels=256, sub_sample=False, bn_layer=True, model_confs=model_confs, mode='Cross', S=6*6)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        
        if self.do_LSTM:
            self.LSTM = nn.LSTM(input_size=4096, hidden_size=3000, num_layers=1, batch_first=True)
            if self.do_Classify:
                self.classifier = nn.Linear(3000, model_confs.num_classes)
        else:
            if self.do_Classify:
                self.classifier = nn.Linear(4096, model_confs.num_classes)
                
                
    def do_BIM(self, x, c=256, w=6, h=6):
        b = x.size(0)//self.T
        x = x.view(b, self.T, c, -1) # x=[b, t, c, h*w]
        x = x.permute(0,2,1,3) # x=[b, c, t, h*w]
        x, f = self.BIM(x)
        x = x.permute(0,2,1,3).contiguous() # x=[b, t, c, h*w]
        x = x.view(b*self.T, c, h, w)
        return x, f
    

    def feature_reshape(self, feas):
        # (N*K, T, fea_size)
        feas = feas.view(feas.size(0)//self.K, self.K, self.T,-1) #(N,K,T,fea_size)
        feas = torch.transpose(feas, 1, 2) #(N,T,K,fea_size)
        feas = feas.contiguous()
        feas = feas.view(feas.size(0)*feas.size(1), -1)
        return feas

    def forward(self, x):
        self.phase = 'trainval' if self.training else 'test'
        self.T = self.num_frames[self.phase]
        x = self.features(x) # [N, 256, 6, 6]
        if self.body_infer:   # x=[N, c, h, w]
            x, BIM_f = self.do_BIM(x, 256, 6, 6)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.fc(x)

        x = x.view(x.size(0)//self.T, self.T, x.size(1))
        cnn_feas = x.contiguous()
        feas = None
        if self.do_LSTM:
            x, (h,c) = self.LSTM(x)
            lstm_feas = x.contiguous()
            # get feas
            feas = torch.cat((cnn_feas, lstm_feas), -1)
        else:
            # get feas
            feas = cnn_feas
        
        if self.do_Classify:
            out = self.classifier(x)
            out = out.view(out.size(0)*out.size(1),-1)
            return feas, out
        else:
            feas = self.feature_reshape(feas)
            return feas, BIM_f


def alexNet(pretrained=False, dataset_name=None, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['alexnet'])
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'classifier' not in k.split('.')[:] and 'LSTM' not in k.split('.')[:]}
        '''for k,v in pretrained_dict.items():
            print k.split('.')[0]'''
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model
