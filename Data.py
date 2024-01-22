import torch.utils.data as data
import glob
import torchvision.transforms as transforms
import PIL
import torch
from natsort import natsorted
class CenterCropLongEdge(object):
    """Crops the given PIL Image on the long edge.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = PIL.Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class Dataset(data.Dataset):
    """
    # -----------------------------------------
    # Get noise/clean for Gaussion denosing
    # -----------------------------------------
    """
    def __init__(self,
                 n_channels=3,
                 H_size=256,
                 path_noise='',
                 path_clean ='',
                 opt='ImageNet',
                 noise_sigma=None):
        super(Dataset, self).__init__()
        self.n_channels = n_channels
        self.patch_size = H_size
        self.noise_sigma= noise_sigma

        self.paths_noise = glob.glob(path_noise+'/*.png')
        self.paths_clean = glob.glob(path_clean+'/*.png')
        if len(self.paths_clean)==0:
            self.paths_noise = glob.glob(path_noise + '/*.PNG')
            self.paths_clean = glob.glob(path_clean + '/*.PNG')
        if len(self.paths_clean)==0:
            self.paths_noise = glob.glob(path_noise + '/*.JPEG')
            self.paths_clean = glob.glob(path_clean + '/*.JPEG')
        if len(self.paths_clean)==0:
            self.paths_noise = glob.glob(path_noise + '/*.tif')
            self.paths_clean = glob.glob(path_clean + '/*.tif')
        
        
        self.paths_clean=natsorted(self.paths_clean)
        self.paths_noise=natsorted(self.paths_noise)        
        self.opt=opt
        
        
        if 'CelebA_HQ' in self.opt:
            self.transform = transforms.Compose([transforms.Resize([self.patch_size, self.patch_size]),
                                                 transforms.ToTensor()])
        elif 'ImageNet' in self.opt:
            self.transform = self.transform = transforms.Compose([CenterCropLongEdge(),
                                                                transforms.Resize(self.patch_size),
                                                                transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):

        # ------------------------------------
        # get clean image
        # ------------------------------------
        clean_path = self.paths_clean[index]
        noise_path = self.paths_noise[index]
        # print(clean_path,noise_path)
        if self.noise_sigma==None:
            img_clean=PIL.Image.open(clean_path).convert('RGB')
            img_noise = PIL.Image.open(noise_path).convert('RGB')

            img_clean = self.transform(img_clean)
            img_noise = self.transform(img_noise)
        else:

            img_clean = PIL.Image.open(clean_path).convert('RGB')
            img_clean = self.transform(img_clean)
            img_noise = img_clean.clone()
            noise = torch.randn(img_clean.size())
            noise = noise/torch.std(noise,unbiased=False)

            noise=noise.mul_(self.noise_sigma/255.0)
            img_noise.add_(noise)

        return img_noise, img_clean

    def __len__(self):
        return len(self.paths_clean)


if __name__ == "__main__":


    x=Dataset(n_channels=3,
            H_size=64,  
            path_noise='./Noisy/',
            path_clean='./GT/',
            opt='test')
