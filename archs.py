import torch
from torch import nn
import timm
from torchvision import models
import torch.nn.functional as F

__all__ = ['VGG11_pre_tr', 'VGG11', 'ResNet34_pre_tr', 'ResNet34', 'ResNet152_pre_tr', 'ResNet152']
    
class GlobalAvgPool2d(nn.Module):
    def __init__(self,
                 device='cpu'):
        super().__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:]).view(-1, x.size(1))
    
class VGG11_pre_tr(nn.Module):
    def __init__(self):
        super(VGG11_pre_tr,self).__init__()
        self.liner_num1= 512
        self.liner_num2= 256
        vgg = timm.create_model('vgg11', pretrained=True)
        self.vgg = nn.Sequential(*list(vgg.children())[:-2])
        self.avg_pool = GlobalAvgPool2d()
        self.classifier = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 12)
        )

    def forward(self,im):
        im = self.vgg(im)
        im = self.avg_pool(im)
        y = self.classifier(im)
        return y
    
class VGG11(nn.Module):
    def __init__(self):
        super(VGG11,self).__init__()
        self.liner_num1= 512
        self.liner_num2= 256
        vgg = timm.create_model('vgg11', pretrained=False)
        self.vgg = nn.Sequential(*list(vgg.children())[:-2])
        self.avg_pool = GlobalAvgPool2d()
        self.classifier = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 12)
        )

    def forward(self,im):
        im = self.vgg(im)
        im = self.avg_pool(im)
        y = self.classifier(im)
        return y
    
class ResNet34_pre_tr(nn.Module):
    def __init__(self):
        super(ResNet34_pre_tr,self).__init__()
        self.liner_num1= 512
        self.liner_num2= 256
        resnet = timm.create_model('resnet34', pretrained=True)
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

class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34,self).__init__()
        self.liner_num1= 512
        self.liner_num2= 256
        resnet = timm.create_model('resnet34', pretrained=False)
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
    
class ResNet152_pre_tr(nn.Module):
    def __init__(self):
        super(ResNet152_pre_tr,self).__init__()
        self.liner_num1= 2048
        self.liner_num2= 256
        resnet = timm.create_model('resnet152', pretrained=True)
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

class ResNet152(nn.Module):
    def __init__(self):
        super(ResNet152,self).__init__()
        self.liner_num1= 2048
        self.liner_num2= 256
        resnet = timm.create_model('resnet152', pretrained=False)
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