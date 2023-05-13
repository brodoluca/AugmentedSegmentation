import torch 
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3,1,1,bias=False),
                nn.BatchNorm2d(out_channels), 
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3,1,1,bias=False),
                nn.BatchNorm2d(out_channels), 
                nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)
    


class UNET_Encoder(nn.Module):
    def __init__(self,
                 in_channels = 3, 
                 features = [64, 128, 256, 512]
                 ):
        super().__init__()
        self.downs = nn.ModuleList()
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature


    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            #print(x.shape)
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        #x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        

        return x, skip_connections
    


class UNET_Decoder(nn.Module):
    def __init__(self,
                    features = [64, 128, 256, 512]
                  ):
            super().__init__()
            self.ups = nn.ModuleList()
           #features = features[:,:,-1]
            for feature in reversed(features):
                self.ups.append(
                        nn.ConvTranspose2d(
                            feature*2, feature, kernel_size=2, stride=2 # double height and width
                        )
                    )
                self.ups.append(DoubleConv(feature*2, feature))
    def forward(self, x, skip_connections):
        for idx in range(0, len(self.ups), 2):
                #print(x.shape)
                x = self.ups[idx](x)
                skip_connection = skip_connections[idx//2]

                if x.shape != skip_connection.shape:
                    x = TF.resize(x, size=skip_connection.shape[2:])
                #print(x.shape)
                contcat_skip = torch.cat((skip_connection, x), dim=1)
                x = self.ups[idx+1](contcat_skip)
        return x

class UNET(nn.Module):
    def __init__(self,
                 in_channels = 3, 
                 out_channels = 1, # depending on this value, you have more segmentation classes
                 features = [64, 128, 256, 512]
                 ):
        super().__init__()

        self.uencoder = UNET_Encoder(in_channels, features)
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.udecoder = UNET_Decoder(features)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    def encode(self, x):
        return self.uencoder(x)
    def forward(self, x):
        x, skip_connections = self.encode(x)
        #print(x.shape)
        x = self.bottleneck(x)
        #print(x.shape)
        x = self.udecoder(x, skip_connections)
        #print(x.shape)
        return self.final_conv(x)

def test():
        x = torch.rand((3,1,161,161))
        model = UNET(in_channels=1, out_channels=1)
        #print(model)
        preds = model(x)
        #print(preds.shape)
        #print(x.shape)


if __name__ == "__main__":
        test()