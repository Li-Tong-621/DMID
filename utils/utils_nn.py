

import torch
import torch.nn as nn
import torch.nn.functional as F

#__________________________________________________________________________________________
#__________________________________________model__________________________________________

class Norm(nn.Module):
    def __init__(self, num_channel, norm_type='batchnorm'):
        super(Norm, self).__init__()
        
        if norm_type == 'batchnorm':
            self.norm = nn.BatchNorm2d(num_channel, affine=True)
        elif norm_type == 'instancenorm':
            self.norm = nn.InstanceNorm2d(num_channel, affine=False)
        elif norm_type == 'none':
            self.norm = nn.Sequential()
        else:
            assert False
    
    def forward(self, x):
        return self.norm(x)

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    
    def __init__(self, in_ch, out_ch, norm='batchnorm'):
        super(double_conv, self).__init__()
        
        self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                Norm(out_ch, norm),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                Norm(out_ch, norm),
                nn.ReLU(inplace=True),
                )
            
    def forward(self, x):
        x = self.conv(x)

        return x
    
class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, norm='batchnorm'):
        super(inconv, self).__init__()
        
        self.conv = double_conv(in_ch, out_ch, norm=norm)

    def forward(self, x):
        x = self.conv(x)

        return x
    
class down(nn.Module):
    def __init__(self, in_ch, out_ch, norm='batchnorm'):
        super(down, self).__init__()

        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch, norm)
        )

    def forward(self, x):
        x = self.mpconv(x)

        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, norm='batchnorm'):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch, norm)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffY // 2, diffY - diffY // 2, 
                        diffX // 2, diffX - diffX // 2), 'replicate')
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        return x

class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, norm='batchnorm'):
        super(up_conv, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch, norm)
        
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)

        return x

class outconv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)

        return x

class UNet(nn.Module):
    def __init__(self, nIn=3, nOut=1, down_sample_norm='instancenorm', 
                 up_sample_norm = 'batchnorm', need_sigmoid=False):
        super(UNet, self).__init__()
        
        self.inc = inconv(nIn, 64, norm=down_sample_norm)
        self.down1 = down(64, 128, norm=down_sample_norm)
        self.down2 = down(128, 256, norm=down_sample_norm)
        self.down3 = down(256, 512, norm=down_sample_norm)
        self.down4 = down(512, 512, norm=down_sample_norm)
        self.up1 = up(1024, 256, norm=up_sample_norm)
        self.up2 = up(512, 128, norm=up_sample_norm)
        self.up3 = up(256, 64, norm=up_sample_norm)
        self.up4 = up(128, 64, norm=up_sample_norm)
        self.outc = outconv(64, nOut)
        
        self.need_sigmoid = need_sigmoid
        
    def forward(self, x):
        self.x1 = self.inc(x)        
        self.x2 = self.down1(self.x1)
        self.x3 = self.down2(self.x2)
        self.x4 = self.down3(self.x3)
        self.x5 = self.down4(self.x4)
        self.x6 = self.up1(self.x5, self.x4)
        self.x7 = self.up2(self.x6, self.x3)
        self.x8 = self.up3(self.x7, self.x2)
        self.x9 = self.up4(self.x8, self.x1)     
        self.y = self.outc(self.x9)
        
        if self.need_sigmoid:
            self.y = torch.sigmoid(self.y)
        
        return self.y
    
    
class Encoder(nn.Module):
    def __init__(self, nIn=3, nOut=3, down_sample_norm='instancenorm', 
                 up_sample_norm='batchnorm'):
        super(Encoder, self).__init__()
        
        self.net = UNet(nIn, 2*nOut,down_sample_norm, up_sample_norm, False)
        
    def forward(self, x):
        x = self.net(x)
        
        mean = x[:, :3]
        log_var = x[:, 3:]
        
        return mean, log_var
    
class Decoder(nn.Module):
    def __init__(self, nIn=3, nOut=3, down_sample_norm='instancenorm', 
                 up_sample_norm='batchnorm'):
        super(Decoder, self).__init__()
        
        self.net = UNet(nIn, nOut,down_sample_norm, up_sample_norm, False)
        
    def forward(self, x):
        
        out = self.net(x)
        
        return out



















#__________________________________________________________________________________________
#__________________________________________image_tools__________________________________________
from PIL import Image
import torch
import numpy as np

def pil_to_np(img_pil, normalize=False):
    img_np = np.array(img_pil)
    if normalize:
        img_np = img_np / 255
        
    if len(img_np.shape) == 2:
        return img_np.astype(np.float)
    else:
        return img_np.transpose(2, 0, 1).astype(np.float)
    
def np_to_pil(img_np, normalize=False):
    if normalize:
        img_np = img_np*255
    
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    
    if len(img_np.shape) == 2:
        img_pil = Image.fromarray(img_np)
    else:
        img_pil = Image.fromarray(img_np.transpose(1, 2, 0))
        
    return img_pil

def np_to_torch(img_np):
    if len(img_np.shape) == 2:
        return torch.Tensor(img_np)[None, None, ...]
    else:
        return torch.Tensor(img_np)[None, ...]

def torch_to_np(img_torch):
    return img_torch.cpu().squeeze().detach().numpy()


