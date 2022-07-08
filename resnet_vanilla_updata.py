import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
# from resnet_decoder_all import *
from resnet_decoder import *
import torch
from torch.autograd import Variable

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

class Flatten(nn.Module):
    def forward(self, inputs):
        return inputs.view(inputs.size(0), -1)
    
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
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
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, num_domains, flags):
        self.inplanes = 64
        self.flags = flags
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.fc = nn.Linear(512, num_classes)
        self.fc_l = nn.Linear(512, num_classes)
        self.fc_h = nn.Linear(512, num_classes)
        self.block6 = nn.Sequential( 
                            nn.AvgPool2d(7),
                            Flatten(),
                            )
       
        self.distangler_H = nn.Sequential(nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
                          nn.BatchNorm2d(512),
                          nn.ReLU(),
                          )
        
        self.distangler_L = nn.Sequential(nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
                          nn.BatchNorm2d(512),
                          nn.ReLU(),
                          )
        
        self.spatial_attention = SpatialAttention(3)
                
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        self.decoder_H = resnet18_decoder()
        self.decoder_L = resnet18_decoder()


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, types):

        end_points = {}
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        out = self.layer4(x)
        if types == 'disentangle':
            L_info = self.distangler_L(out)
            x_L_image = self.decoder_L(L_info)
            H_info = self.distangler_H(out)
            x_H_image = self.decoder_H(H_info)
            l = self.block6(L_info)
            h = self.block6(H_info)
            x_l = self.fc_l(l)
            x_h = self.fc_h(h)
            
            return x_l, x_h, x_L_image, x_H_image
        
        elif types == 'interact':  
            L_info = self.distangler_L(out)
            L_att = self.spatial_attention(L_info)
            H_info = self.distangler_H(out)
            
            inter_info = H_info*L_att
            inter_info = self.block6(inter_info)
            cls = self.fc(inter_info)
                        
            end_points['Predictions'] = F.softmax(input=cls, dim=-1)
            return cls, end_points

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = model_zoo.load_url(model_urls['resnet18'])

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               k in model_dict and v.size() == model_dict[k].size()}

        print('model dict keys:', len(model_dict.keys()), 'pretrained keys:', len(pretrained_dict.keys()))
        print('model dict keys:', model_dict.keys(), 'pretrained keys:', pretrained_dict.keys())
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    return model

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               k in model_dict and v.size() == model_dict[k].size()}

        print('model dict keys:', len(model_dict.keys()), 'pretrained keys:', len(pretrained_dict.keys()))
        print('model dict keys:', model_dict.keys(), 'pretrained keys:', pretrained_dict.keys())
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    return model
