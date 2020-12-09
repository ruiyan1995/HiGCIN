import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from .Mask import Mask

class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True, model_confs=None, mode=None, S=None):
        super(_NonLocalBlockND, self).__init__()
#         self.RelationMap = Mask(mode, T=model_confs.used_frames, S=S).get_RelationMap()
        ###
        phases = ['trainval', 'test']
        self.RelationMaps = {phase: Mask(mode, T=model_confs.num_frames[phase], S=S).get_RelationMap() for phase in phases}
        
        assert dimension in [1, 2, 3]
#         self.g_function = 'Linear'
        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        
#         if self.g_function == 'Linear':
#             self.g = nn.Linear(self.in_channels, self.inter_channels)
#         else:
#             self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
#                          kernel_size=1, stride=1, padding=0)
            
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        phase = 'trainval' if self.training else 'test'
        self.cur_RelationMap = self.RelationMaps[phase]
        # x: (b, c, t, h*w)
        b, c = x.size(0), x.size(1)

        g_x = self.g(x).view(b, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        
        
#         if self.g_function == 'Linear':
#             _x = x.contiguous().view(b, c, -1) #(b, c, t*h*w)
#             _x = _x.permute(0, 2, 1) #(b, t*h*w, c)
#             g_x = self.g(_x)
#         else:
#             g_x = self.g(x).view(b, self.inter_channels, -1)
#             g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(b, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(b, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
#         print(f.size(), self.cur_RelationMap.repeat(f.size(0), 1, 1).size())
        if torch.cuda.is_available():
            f = f*self.cur_RelationMap.repeat(f.size(0), 1, 1).cuda() #(b,t*k,t*k) !!!!!!!!!!.cuda()!!!!!!!!
        else:
            f = f*self.cur_RelationMap.repeat(f.size(0), 1, 1)
        N = len(f[0,0,:].nonzero()) # k
        #N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(b, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z, f


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True, model_confs=None, mode=None):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer, model_confs=model_confs,
                                            mode=mode)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True, model_confs=None, mode=None, S=None):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer, model_confs=model_confs,
                                            mode=mode, S=S)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True, model_confs=None, mode=None):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer, model_confs=model_confs,
                                            mode=mode)