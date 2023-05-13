import torch
import torch.nn as nn
from torch.autograd import Variable

from encoder import PointNetfeat
from UNET import UNET
import config



class ConvBlock(nn.Module):
    def __init__(self, in_channels,out_channels, kernel = (1,1) ):
        super().__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel,padding=(0,0),bias=True),
                nn.BatchNorm2d(out_channels), 
                nn.ReLU(inplace=True),
                #nn.Conv2d(out_channels, out_channels, kernel,1,1,bias=True),
                #nn.BatchNorm2d(out_channels), 
                #nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)


class PointEncoder(nn.Module):
    def __init__(self, out_channels = 3, concat_channel = config.FOR_REFLECTANCE):

        super().__init__()
        self.out_channels = out_channels
        self.concat_channel = concat_channel
        self.first_convs = nn.ModuleList(
                [ConvBlock(3, 16, 1), 
                ConvBlock(16, 64, 1), 
                ConvBlock(64, 128, 1)]
        )
        self.maxpool = nn.MaxPool2d((8,1),(8,1))
        self.conv4 = ConvBlock(132, 64, 1)
        self.conv5 = ConvBlock(64, 16, 1)
        self.conv6 = ConvBlock(16, out_channels, 1)
        
        
    def forward(self, points, neighbours):
        out = neighbours[:,:3, :, :]
        #print(out.shape)
        for conv in self.first_convs:
            out = conv(out)
         #   print(out.shape)
        #nn.MaxPool2d()
        out = out.permute(0,3,2,1)
        out = self.maxpool(out)
        #print(out.shape)
        out = out.permute(0,3,2,1)
        #print(out.shape)
        #print(points.shape)
        #here it should be concat to the poi
        # nts
            #x = tf.concat([x, points[...,0:4]], axis=3, name='pn_concat')

        if(self.concat_channel == config.FOR_REFLECTANCE):
            out = torch.cat(( out, points[:, 0:4,:, :]), dim = 1)
        else:
            temp = points[:, :,0:3, :]
            temp = torch.cat(temp, points[:, :,5, :])
            out = torch.cat(out, temp, dim = 1)
        #print("after concat",out.shape)

        
        out = self.conv4(out)
        #print(out.shape)
        out = self.conv5(out)
        #print(out.shape)
        out = self.conv6(out)
        #print(out.shape)
        out = out.view(-1, self.out_channels,config.IMAGE_HEIGHT, config.IMAGE_WIDTH)

        return out

class LuNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.pointnet = PointEncoder()
        self.unet = UNET(out_channels = n_classes)
        
    def forward(self, p, n):
        out = self.pointnet(p,n)
        #print(out.shape)
        out = self.unet(out)
        #print(out.shape)
        return out




if __name__ == "__main__":
    points = Variable(torch.rand(32,4,1,config.IMAGE_HEIGHT*config.IMAGE_WIDTH))
    neighbours = Variable(torch.rand(32,4,8,config.IMAGE_HEIGHT*config.IMAGE_WIDTH))

    lunet = LuNet(3)

    out = lunet(points,neighbours)
    print('point feat', out.size())