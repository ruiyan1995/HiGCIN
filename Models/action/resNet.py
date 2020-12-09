import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from Models.activity.lib.non_local_dot_product import NONLocalBlock2D
import torch.nn.functional as F
__all__ = ['ResNet', 'resNet18', 'resNet34', 'resNet50', 'resNet101',
           'resNet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False, model_confs=None, body_infer=True, body_infer_mode='Cross', do_LSTM=True, do_Classify=True):
        super(ResNet, self).__init__()
        self.do_LSTM = do_LSTM
        self.do_Classify = do_Classify
        self.body_infer = body_infer
        self.body_infer_mode = body_infer_mode
        
        self.num_frames = model_confs.num_frames
        self.K = model_confs.num_players
        self.num_classes = model_confs.num_classes
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        ###########################################################################################
        if self.body_infer:
            self.BIM = NONLocalBlock2D(in_channels=512, sub_sample=False, bn_layer=True, model_confs=model_confs, mode=body_infer_mode, S=7*7)

        
        if self.do_LSTM:
            self.LSTM = nn.LSTM(input_size=512 * block.expansion, hidden_size=3000, num_layers=1, batch_first=True)
            if self.do_Classify:
                self.classifier = nn.Linear(3000, self.num_classes)
        else:
            if self.do_Classify:
                self.classifier = nn.Linear(512 * block.expansion, self.num_classes)
                
        ###########################################################################################

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def do_BIM(self, x, c, w, h):
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        
        ################################################
        #print x.size()
        if self.body_infer:   # x=[N, c, h, w]
            x, BIM_f = self.do_BIM(x, 512, 7, 7)
            #x = F.dropout(x, training=self.training)
        ################################################
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        ####################################################
        x = x.view(x.size(0)//self.T, self.T, x.size(-1))
        # get feas
        cnn_feas = x.contiguous()

        #print 'cnn_feas:', cnn_feas.size()
        feas = None
        if self.do_LSTM:
            x, (h, c) = self.LSTM(x)
            # get feas
            lstm_feas = x.contiguous()
            #print 'lstm_feas:', lstm_feas.size()
            feas = torch.cat((cnn_feas, lstm_feas), -1)
        else:
            # get feas
            feas = cnn_feas
        
        if self.do_Classify:
            out = self.classifier(x)
            out = out.view(out.size(0)*out.size(1),-1)
            return feas, out
        else:
            ################################################
            feas = self.feature_reshape(feas)
            ################################################
            return feas, BIM_f



def resNet18(pretrained=False, dataset_name=None, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'fc' not in k.split('.')[:] and 'classifier' not in k.split('.')[:] and 'LSTM' not in k.split('.')[:]}
           
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def resNet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = model_zoo.load_url(model_urls['resnet34'])
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k.split('.')[
                0] != 'fc'}

        #for k,v in pretrained_dict.items():
        #    print k.split('.')[0]
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def resNet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k.split('.')[
                0] != 'fc'}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def resNet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resNet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
