import torch.nn as nn
import torch.nn.functional as F


class UnFlatten(nn.Module):
    def forward(self,input):
        return input.view(input.size(0),512,4,4)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def Upsample2d(in_planes, out_planes, stride=2):
    return nn.Upsample(scale_factor=2, mode="nearest")


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, flag=True):
        super(BasicBlock, self).__init__()
        self.conv0_0 = nn.Conv2d(inplanes,planes,3,padding=1)
        self.bn0_0 = nn.BatchNorm2d(planes)
        
        self.conv0_1 = nn.Conv2d(inplanes,inplanes,3,padding=1)
        self.bn0_1 = nn.BatchNorm2d(inplanes)
        
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Upsample2d(inplanes, inplanes)
        self.bn2 = nn.BatchNorm2d(inplanes)
        self.upsample = upsample
        self.stride = stride
        self.flag = flag

    def forward(self, x):
        residual = x
        if self.upsample is not None:
            residual_temp = self.upsample(x)
            residual = self.conv0_0(residual_temp)
            residual = self.bn0_0(residual)

        if self.flag:
            x = self.conv2(x)
            x = self.conv0_1(x)
            x = self.bn0_1(x)
            
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        out = x + residual

        return out


class ResNetDecoder(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 512
        super(ResNetDecoder, self).__init__()
        
        self.conv_end = nn.Conv2d(64,3,kernel_size=3,stride=1,padding=1)
            
        self.layer0 = self._make_layer(block, 512, layers[0])
        self.layer1 = self._make_layer(block, 256, layers[1])
        self.layer2 = self._make_layer(block, 128, layers[2])
        self.layer3 = self._make_layer(block, 64, layers[3])
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def bn_eval(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _make_layer(self, block, planes, blocks, stride=1, flag=False):
        upsample = nn.Sequential(
                Upsample2d(self.inplanes, planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample=upsample, flag=True))
        for i in range(1, blocks):
            layers.append(block(planes, planes, flag=False))
            self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
                
        x = self.layer0(x) #B×512×14×14
        x = self.layer1(x) #B×256×28×28
        x = self.layer2(x) #B×128×56×56
        x = self.layer3(x) #B×64×112×112

        x = self.conv_end(x)
        out = self.sigmoid(x)

        return out

def resnet18_decoder(pretrained=False):
    """
    Constructs a ResNet-18 decoder model.
    """
    model = ResNetDecoder(BasicBlock, [2, 2, 2, 2])
    return model

