import torch
from torch import nn
from torch.nn import functional as F


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(128)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm3 = nn.BatchNorm2d(256)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm4 = nn.BatchNorm2d(512)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.relu1(self.norm1(self.conv1(x)))
        out = self.relu2(self.norm2(self.conv2(out)))
        out = self.relu3(self.norm3(self.conv3(out)))
        out = self.relu4(self.norm4(self.conv4(out)))
        out = self.sigmoid(self.conv5(out))
        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.conv1 = nn.ConvTranspose2d(100, 512, kernel_size=4, stride=1, padding=0, bias=False) # 4x4
        self.norm1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False) # 8x8
        self.norm2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False) # 16x16
        self.norm3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False) # 32x32
        self.norm4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        
        self.conv5 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False) # 64x64
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        out = self.relu1(self.norm1(self.conv1(x)))
        out = self.relu2(self.norm2(self.conv2(out)))
        out = self.relu3(self.norm3(self.conv3(out)))
        out = self.relu4(self.norm4(self.conv4(out)))
        out = self.tanh(self.conv5(out))
        return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
