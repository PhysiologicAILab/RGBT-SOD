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

        
        #self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.resnet = models.resnet50(pretrained=True)
        
        self.resnet_thermal = models.resnet50(pretrained=True)
        self.convt0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)

                 
        self.initial_out = nn.Conv2d(2048, 3, kernel_size=1, padding=0)
        self.initial_out_thermal = nn.Conv2d(2048, 3, kernel_size=1, padding=0)
        
        self.fuse = nn.Conv2d(2048,512,kernel_size=3,padding=1)
        
        self.fuse_out = nn.Conv2d(512,3,kernel_size=1,padding=0)
        

        
        
        
        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

    def forward(self, x,thermal):
        input = x
        
        x = self.resnet.conv1(x)
        
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)      # bs, 64, 88, 88
        x1 = self.resnet.layer1(x)      # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44

        x3 = self.resnet.layer3(x2)     # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)
        
        t = self.convt0(thermal)
        t = self.resnet_thermal.bn1(t)
        t = self.resnet_thermal.relu(t)
        t = self.resnet_thermal.maxpool(t)      # bs, 64, 88, 88
        t1 = self.resnet_thermal.layer1(t)      # bs, 256, 88, 88
        t2 = self.resnet_thermal.layer2(t1)     # bs, 512, 44, 44

        t3 = self.resnet_thermal.layer3(t2)     # bs, 1024, 22, 22
        t4 = self.resnet_thermal.layer4(t3)

        
        output_final = F.interpolate(self.initial_out(x4), input.size()[2:], mode="bilinear")
        output_thermal = F.interpolate(self.initial_out_thermal(t4), input.size()[2:], mode="bilinear")
        
        fusion = self.fuse(x4+t4)
        output_fuse = F.interpolate(self.fuse_out(fusion), input.size()[2:], mode="bilinear")

        if self.training:
            return output_final

        return output_fuse

    



if __name__ == "__main__":
    model = RGBD_sal()
    model.cuda()
    input = torch.autograd.Variable(torch.zeros(4, 3, 256, 256)).cuda()
    depth = torch.autograd.Variable(torch.zeros(4, 1, 256, 256)).cuda()
    output = model(input, depth)
    #print(output.size())
