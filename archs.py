import torch
from torch import nn
import timm
from torchvision import models
import torch.nn.functional as F

__all__ = ['Effnet', 'ResNet18', 'ResNet101']
    
class GlobalAvgPool2d(nn.Module):
    def __init__(self,
                 device='cpu'):
        super().__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:]).view(-1, x.size(1))
    
    
class Effnet(nn.Module):
    def __init__(self):
        super(Effnet,self).__init__()
        effnet = timm.create_model('efficientnet_b0', pretrained=True)
        self.effnet = nn.Sequential(*list(effnet.children())[:-2])
        self.fc1 = nn.Linear(1280, 12)
        self.avg_pool = GlobalAvgPool2d()
        
    def forward(self,im):
        im = self.effnet(im)
        im = self.avg_pool(im)
        y = self.fc1(im)
        y = torch.log_softmax(y, dim=-1)
        return y
    
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18,self).__init__()
        self.liner_num1= 512
        self.liner_num2= 256
        resnet = timm.create_model('resnet18', pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.avg_pool = GlobalAvgPool2d()
        self.classifier = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 12)
        )

    def forward(self,im):
        im = self.resnet(im)
        im = self.avg_pool(im)
        y = self.classifier(im)
        return y
    
    
class ResNet101(nn.Module):
    def __init__(self):
        super(ResNet101,self).__init__()
        self.liner_num1= 2048
        self.liner_num2= 256
        resnet = timm.create_model('resnet101', pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.avg_pool = GlobalAvgPool2d()
        self.classifier = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 12)
        )

    def forward(self,im):
        im = self.resnet(im)
        im = self.avg_pool(im)
        y = self.classifier(im)
        return y