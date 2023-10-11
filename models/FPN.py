import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torchvision import models
import torch.nn.init as init
from torch.nn import Conv2d, Parameter, Softmax
from functools import partial
import matplotlib.pyplot as plt
import math


    
class RGBD_sal_transformer(nn.Module):
    def __init__(self):
        super(RGBD_sal_transformer, self).__init__()

        
        feats = list(models.vgg16_bn(pretrained=True).features.children())
        self.conv0 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1 = nn.Sequential(*feats[1:6])
        self.conv2 = nn.Sequential(*feats[6:13])
        self.conv3 = nn.Sequential(*feats[13:23])
        self.conv4 = nn.Sequential(*feats[23:33])
        self.conv5 = nn.Sequential(*feats[33:43])
        
        
        feats_t = list(models.vgg16_bn(pretrained=True).features.children())
        self.conv0t = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv1t = nn.Sequential(*feats_t[1:6])
        self.conv2t = nn.Sequential(*feats_t[6:13])
        self.conv3t = nn.Sequential(*feats_t[13:23])
        self.conv4t = nn.Sequential(*feats_t[23:33])
        self.conv5t = nn.Sequential(*feats_t[33:43])
         
        self.merge4 = nn.Conv2d(512, 512, kernel_size=1, padding=0)
        self.merge3 = nn.Conv2d(512, 256, kernel_size=1, padding=0)
        self.merge2 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.merge1 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        
        self.f4_ouput = nn.Conv2d(512, 5, kernel_size=1, padding=0)
        self.f3_ouput = nn.Conv2d(256, 5, kernel_size=1, padding=0)
        self.f2_ouput = nn.Conv2d(128, 5, kernel_size=1, padding=0)
        self.f1_ouput = nn.Conv2d(64, 5, kernel_size=1, padding=0)
                 
        #self.initial_out = nn.Conv2d(512, 5, kernel_size=1, padding=0)
        
        #self.initial_out_t = nn.Conv2d(512, 5, kernel_size=1, padding=0)
        

        
        
        
        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

    def forward(self, x,thermal):
        input = x
        

        
        c0 = self.conv0(x)
        c1 = self.conv1(c0)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        
        t0 = self.conv0t(thermal)
        t1 = self.conv1t(t0)
        t2 = self.conv2t(t1)
        t3 = self.conv3t(t2)
        t4 = self.conv4t(t3)
        t5 = self.conv5t(t4)
        

        
        f5 = c5+t5
        
        f4 = c4  + F.interpolate(self.merge4(f5), c4.size()[2:], mode="bilinear") + t4
        
        
        f3 = c3  + F.interpolate(self.merge3(f4), c3.size()[2:], mode="bilinear") + t3
        
        
        f2 = c2  + F.interpolate(self.merge2(f3), c2.size()[2:], mode="bilinear") + t2
        
        
        f1 = c1  + F.interpolate(self.merge1(f2), c1.size()[2:], mode="bilinear") + t1
        
        
        f4_attention =F.interpolate(self.f4_ouput(f4), input.size()[2:], mode="bilinear")
        f3_attention =F.interpolate(self.f3_ouput(f3), input.size()[2:], mode="bilinear")
        f2_attention =F.interpolate(self.f2_ouput(f2), input.size()[2:], mode="bilinear")
        f1_attention =F.interpolate(self.f1_ouput(f1), input.size()[2:], mode="bilinear")
        
        #initial_out = F.interpolate(self.initial_out(c5), input.size()[2:], mode="bilinear")
        #initial_out_t = F.interpolate(self.initial_out_t(t5), input.size()[2:], mode="bilinear")
        
        

        if self.training:
            return f4_attention,f3_attention,f2_attention,f1_attention#,ct,t5

        return f1_attention

    



if __name__ == "__main__":
    model = RGBD_sal()
    model.cuda()
    input = torch.autograd.Variable(torch.zeros(4, 3, 256, 256)).cuda()
    depth = torch.autograd.Variable(torch.zeros(4, 1, 256, 256)).cuda()
    output = model(input, depth)
    #print(output.size())
