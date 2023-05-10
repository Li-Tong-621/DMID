

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
#__________________________________________image_io__________________________________________

import glob

import torch
import torchvision
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.use('agg')


def crop_image(img, d=32):
    """
    Make dimensions divisible by d

    :param pil img:
    :param d:
    :return:
    """

    new_size = (img.size[0] - img.size[0] % d,
                img.size[1] - img.size[1] % d)

    bbox = [
        int((img.size[0] - new_size[0]) / 2),
        int((img.size[1] - new_size[1]) / 2),
        int((img.size[0] + new_size[0]) / 2),
        int((img.size[1] + new_size[1]) / 2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped


def crop_np_image(img_np, d=32):
    return torch_to_np(crop_torch_image(np_to_torch(img_np), d))


def crop_torch_image(img, d=32):
    """
    Make dimensions divisible by d
    image is [1, 3, W, H] or [3, W, H]
    :param pil img:
    :param d:
    :return:
    """
    new_size = (img.shape[-2] - img.shape[-2] % d,
                img.shape[-1] - img.shape[-1] % d)
    pad = ((img.shape[-2] - new_size[-2]) // 2, (img.shape[-1] - new_size[-1]) // 2)

    if len(img.shape) == 4:
        return img[:, :, pad[-2]: pad[-2] + new_size[-2], pad[-1]: pad[-1] + new_size[-1]]
    assert len(img.shape) == 3
    return img[:, pad[-2]: pad[-2] + new_size[-2], pad[-1]: pad[-1] + new_size[-1]]


def get_params(opt_over, net, net_input, downsampler=None):
    """
    Returns parameters that we want to optimize over.
    :param opt_over: comma separated list, e.g. "net,input" or "net"
    :param net: network
    :param net_input: torch.Tensor that stores input `z`
    :param downsampler:
    :return:
    """

    opt_over_list = opt_over.split(',')
    params = []

    for opt in opt_over_list:

        if opt == 'net':
            params += [x for x in net.parameters()]
        elif opt == 'down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'

    return params


def get_image_grid(images_np, nrow=8):
    """
    Creates a grid from a list of images by concatenating them.
    :param images_np:
    :param nrow:
    :return:
    """
    images_torch = [torch.from_numpy(x).type(torch.FloatTensor) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)

    return torch_grid.numpy()


def plot_image_grid(name, images_np, interpolation='lanczos', output_path="output/"):
    """
    Draws images in a grid

    Args:
        images_np: list of images, each image is np.array of size 3xHxW or 1xHxW
        nrow: how many images will be in one row
        interpolation: interpolation used in plt.imshow
    """
    assert len(images_np) == 2 
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"

    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = get_image_grid(images_np, 2)

    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)

    plt.savefig(output_path + "{}.png".format(name))


def save_image(name, image_np, output_path="output/"):
    p = np_to_pil(image_np)
    p.save(output_path + "{}.jpg".format(name))

def video_to_images(file_name, name):
    video = prepare_video(file_name)
    for i, f in enumerate(video):
        save_image(name + "_{0:03d}".format(i), f)

def images_to_video(images_dir ,name, gray=True):
    num = len(glob.glob(images_dir +"/*.jpg"))
    c = []
    for i in range(num):
        if gray:
            img = prepare_gray_image(images_dir + "/"+  name +"_{}.jpg".format(i))
        else:
            img = prepare_image(images_dir + "/"+name+"_{}.jpg".format(i))
        print(img.shape)
        c.append(img)
    save_video(name, np.array(c))

def save_heatmap(name, image_np):
    cmap = plt.get_cmap('jet')

    rgba_img = cmap(image_np)
    rgb_img = np.delete(rgba_img, 3, 2)
    save_image(name, rgb_img.transpose(2, 0, 1))


def save_graph(name, graph_list, output_path="output/"):
    plt.clf()
    plt.plot(graph_list)
    plt.savefig(output_path + name + ".png")


def create_augmentations(np_image):
    """
    convention: original, left, upside-down, right, rot1, rot2, rot3
    :param np_image:
    :return:
    """
    aug = [np_image.copy(), np.rot90(np_image, 1, (1, 2)).copy(),
           np.rot90(np_image, 2, (1, 2)).copy(), np.rot90(np_image, 3, (1, 2)).copy()]
    flipped = np_image[:,::-1, :].copy()
    aug += [flipped.copy(), np.rot90(flipped, 1, (1, 2)).copy(), np.rot90(flipped, 2, (1, 2)).copy(), np.rot90(flipped, 3, (1, 2)).copy()]
    return aug


def create_video_augmentations(np_video):
    """
        convention: original, left, upside-down, right, rot1, rot2, rot3
        :param np_video:
        :return:
        """
    aug = [np_video.copy(), np.rot90(np_video, 1, (2, 3)).copy(),
           np.rot90(np_video, 2, (2, 3)).copy(), np.rot90(np_video, 3, (2, 3)).copy()]
    flipped = np_video[:, :, ::-1, :].copy()
    aug += [flipped.copy(), np.rot90(flipped, 1, (2, 3)).copy(), np.rot90(flipped, 2, (2, 3)).copy(),
            np.rot90(flipped, 3, (2, 3)).copy()]
    return aug


def save_graphs(name, graph_dict, output_path="output/"):
    """

    :param name:
    :param dict graph_dict: a dict from the name of the list to the list itself.
    :return:
    """
    plt.clf()
    fig, ax = plt.subplots()
    for k, v in graph_dict.items():
        ax.plot(v, label=k)
        # ax.semilogy(v, label=k)
    ax.set_xlabel('iterations')
    # ax.set_ylabel(name)
    ax.set_ylabel('MSE-loss')
    # ax.set_ylabel('PSNR')
    plt.legend()
    plt.savefig(output_path + name + ".png")


def load(path):
    """Load PIL image."""
    img = Image.open(path)
    return img


def get_image(path, imsize=-1):
    """Load an image and resize to a cpecific size.

    Args:
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = load(path)

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0] != -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)

    img_np = pil_to_np(img)

    return img, img_np


def prepare_image(file_name):
    """
    loads makes it divisible
    :param file_name:
    :return: the numpy representation of the image
    """
    img_pil = crop_image(get_image(file_name, -1)[0], d=32)
    return pil_to_np(img_pil)


def prepare_gray_image(file_name):
    img = prepare_image(file_name)
    return np.array([np.mean(img, axis=0)])


def pil_to_np(img_PIL, with_transpose=True):
    """
    Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    """
    ar = np.array(img_PIL)
    if len(ar.shape) == 3 and ar.shape[-1] == 4:
        ar = ar[:, :, :3]
        # this is alpha channel
    if with_transpose:
        if len(ar.shape) == 3:
            ar = ar.transpose(2, 0, 1)
        else:
            ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def median(img_np_list):
    """
    assumes C x W x H [0..1]
    :param img_np_list:
    :return:
    """
    assert len(img_np_list) > 0
    l = len(img_np_list)
    shape = img_np_list[0].shape
    result = np.zeros(shape)
    for c in range(shape[0]):
        for w in range(shape[1]):
            for h in range(shape[2]):
                result[c, w, h] = sorted(i[c, w, h] for i in img_np_list)[l//2]
    return result


def average(img_np_list):
    """
    assumes C x W x H [0..1]
    :param img_np_list:
    :return:
    """
    assert len(img_np_list) > 0
    l = len(img_np_list)
    shape = img_np_list[0].shape
    result = np.zeros(shape)
    for i in img_np_list:
        result += i
    return result / l


def np_to_pil(img_np):
    """
    Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    :param img_np:
    :return:
    """
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        assert img_np.shape[0] == 3, img_np.shape
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


def np_to_torch(img_np):
    """
    Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]

    :param img_np:
    :return:
    """
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    """
    Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """
    return img_var.detach().cpu().numpy()[0]


