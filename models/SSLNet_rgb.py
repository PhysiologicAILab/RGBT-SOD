import torch
import torch.nn as nn
import torch.functional as F
from models.Res2Net_v1b import res2net50_v1b_26w_4s


class HFM(nn.Module):

    def __init__(self, num_channels,deep_channels,reduction_ratio=2):

        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """

        super(HFM, self).__init__()

        
        num_channels_reduced = 128

        self.reduction_ratio = reduction_ratio

        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)

        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        
        #self.transfer = nn.Conv2d(num_channels,1,1)
        

        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()



    def forward(self, input_tensor,rgb):

        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """      

        batch_size, num_channels, H, W = input_tensor.size()

        # Average along each channel
        
        squeeze_tensor = rgb.view(batch_size, num_channels, -1).mean(dim=2)

        fc_out_1 = self.relu(self.fc1(squeeze_tensor))

        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1)) + rgb

        return output_tensor




class SSLNet(nn.Module):
    def __init__(self):
        super(SSLNet, self).__init__()

        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        

        # Upsample_model
        
        
        self.rgbconv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.rgbconv2 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.rgbconv3 = nn.Conv2d(512, 64, kernel_size=3, padding=1)
        self.rgbconv4 = nn.Conv2d(1024, 64, kernel_size=3, padding=1)
        self.rgbconv5 = nn.Conv2d(2048, 64, kernel_size=3, padding=1)
        

        
        
        self.channel1 = HFM(64,64)
        self.channel2 = HFM(256,256)
        self.channel3 = HFM(512,512)
        self.channel4 = HFM(1024,1024)
        self.channel5 = HFM(2048,2048)
        


        # Upsample_model
        
        self.upsample1_g = nn.Sequential(nn.Conv2d(128, 32, 3, 1, 1, ), nn.BatchNorm2d(32), nn.GELU(),
                                         nn.UpsamplingBilinear2d(scale_factor=4, ))

        self.upsample2_g = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1, ), nn.BatchNorm2d(64), nn.GELU(),
                                         )

        self.upsample3_g = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1, ), nn.BatchNorm2d(64), nn.GELU(),
                                         nn.UpsamplingBilinear2d(scale_factor=2, ))

        self.upsample4_g = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1, ), nn.BatchNorm2d(64), nn.GELU(),
                                         nn.UpsamplingBilinear2d(scale_factor=2, ))
        

        self.upsample5_g = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, ), nn.BatchNorm2d(64), nn.GELU(),
                                         nn.UpsamplingBilinear2d(scale_factor=2, ))


        self.conv_g = nn.Conv2d(32, 1, 1)
        self.convr_g = nn.Conv2d(2048, 1, 1)
        self.convt_g = nn.Conv2d(2048, 1, 1)



    def forward(self, rgb):


        A1 = self.resnet.conv1(rgb)
        A1 = self.resnet.bn1(A1)
        A1 = self.resnet.relu(A1)
        A1 = self.resnet.maxpool(A1)      # bs, 64, 88, 88
        A2 = self.resnet.layer1(A1)      # bs, 256, 88, 88
        A3 = self.resnet.layer2(A2)     # bs, 512, 44, 44
        A4 = self.resnet.layer3(A3)     # bs, 1024, 22, 22
        A5 = self.resnet.layer4(A4)
        '''
        A1_t = self.resnet.conv1(ti)
        A1_t = self.resnet.bn1(A1_t)
        A1_t = self.resnet.relu(A1_t)
        A1_t = self.resnet.maxpool(A1_t)      # bs, 64, 88, 88
        A2_t = self.resnet.layer1(A1_t)      # bs, 256, 88, 88
        A3_t = self.resnet.layer2(A2_t)     # bs, 512, 44, 44
        A4_t = self.resnet.layer3(A3_t)     # bs, 1024, 22, 22
        A5_t = self.resnet.layer4(A4_t)
        '''
            
            
        F5 = self.channel5(A5,A5) 
        F4 = self.channel4(A4,A4) 
        F3 = self.channel3(A3,A3) 
        F2 = self.channel2(A2,A2)
        F1 = self.channel1(A1,A1)
        
        

        
        F5 = self.rgbconv5(F5)
        F4 = self.rgbconv4(F4)
        F3 = self.rgbconv3(F3)
        F2 = self.rgbconv2(F2)
        F1 = self.rgbconv1(F1)
        


        
        


        F5 = self.upsample5_g(F5)

        F4 = torch.cat((F4, F5), dim=1)     
        F4 = self.upsample4_g(F4)

        F3 = torch.cat((F3, F4), dim=1)
        F3 = self.upsample3_g(F3)

        F2 = torch.cat((F2, F3), dim=1)
        F2 = self.upsample2_g(F2)

        F1 = torch.cat((F1, F2), dim=1)
        F1 = self.upsample1_g(F1)
        
        
        
        out = torch.nn.functional.interpolate(self.conv_g(F1), rgb.size()[2:], mode="bilinear")

        
        if self.training:

            out_a5 = torch.nn.functional.interpolate(self.convr_g(A5), rgb.size()[2:], mode="bilinear")
            out_a5_t = torch.nn.functional.interpolate(self.convt_g(A5), rgb.size()[2:], mode="bilinear")

            return out,out_a5,out_a5_t
            

        return out
