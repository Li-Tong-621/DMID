import os
import time
import glob
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from guided_diffusion.unet import UNetModel
import utils.utils_image as util
from Data import Dataset
import utils.utils_sampling as utils_sampling
from collections import OrderedDict
import os
import torch.nn.functional as F
import lpips
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# loss_fn =lpips.LPIPS(net='alex', version='0.1').to('cuda')
from natsort import natsorted
import torchvision.transforms as transforms
import PIL
from utils.utils_cal_N_2 import find_N
def data_transform(X):
    return 2 * X - 1.0
    #return X

def data_transform_reverse(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)
    #return X

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def MeanUpsample(x, scale):
    n, c, h, w = x.shape
    out = torch.zeros(n, c, h, scale, w, scale).to(x.device) + x.view(n, c, h, 1, w, 1)
    out = out.view(n, c, scale * h, scale * w)
    return out

class Train(object):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device_id = config['device_id']
        self.device = torch.device(self.device_id)

        self.model = UNetModel()
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(self.config['save']['ddpm_checkpoint'], map_location=self.device))

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)

        betas = get_beta_schedule(
            beta_schedule=config['diffusion']['beta_schedule'],
            beta_start=config['diffusion']['beta_start'],
            beta_end=config['diffusion']['beta_end'],
            num_diffusion_timesteps=config['diffusion']['num_diffusion_timesteps'], )

        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        dataset_test = None

        self.val_loader = None
        print('____________________________prepare_over____________________________')

    def sample_image(self,
                     x,
                     last=True,
                     eta=0.8,
                     difussion_times=500
                     ):
        #______________________embedding for gaussian___________________

        e = torch.randn_like(x[:, :, :, :])
        t = torch.ones(x.shape[0]).to(self.device)
        t = t*(difussion_times-1)
        t = torch.tensor(t,dtype=torch.int)
        a = (1 - self.betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        x_N = x[:, :, :, :] * a.sqrt()
        #____________________________  end  ____________________________

        
        skip = difussion_times // self.config['diffusion']['sampling_timesteps']
        seq = range(0, difussion_times, skip)
        seq=list(seq[:-1])+[difussion_times-1]  #a liilte different from ddim


        xs = utils_sampling.generalized_steps(x=x_N,
                                            seq=seq,
                                            model=self.model,
                                            b=self.betas,
                                            eta=eta)

        if last:
            xs = xs[0][-1]
        return xs

    def eval_ddpm(self,
                  eta=0.8,
                  eval_time=1,
                  difussion_times=500,
                  ):

        avg_lpips=0.0
        avg_rgb_psnr = 0.0
        avg_rgb_ssim = 0.0

        idx = 0
        
        cleans = natsorted(glob.glob(self.config['data']['test_path_clean'] + '*.png')+glob.glob(self.config['data']['test_path_clean'] + '*.JPG'))
        #cleans = sorted(glob.glob(self.config['data']['test_path_clean'] + '*mean.JPG'))
        noises = torch.load(self.config['data']['test_path_noise'], map_location='cpu')
        
        noiselevel=[]
        trans=transforms.ToTensor()
        with torch.no_grad():
           
            for index in range(len(cleans)):
                
                noise=noises[str(index)]['noise_image']/255
                noise_level= noises[str(index)]['noise_level']

                #CC:
                if 'CC' in self.config['data']['test_path_clean']:
                    difussion_times=21
                #polyu:
                if 'PolyU' in self.config['data']['test_path_clean']:
                    difussion_times=9
                #fmdd
                
                if 'FMDD' in self.config['data']['test_path_clean']:
                    difussion_times=33
                    difussion_times= find_N(sigma=2* noises[str(index)]['noise_level'] / 255)
                    noise=noise.repeat(1,3,1,1)
                
                clean=PIL.Image.open(cleans[index])
                clean=trans(clean)
                clean=torch.unsqueeze(clean, 0)

                data_start = time.time()
                b, c, h, w = noise.shape

                clean = clean.to(self.device)
                noise = noise.to(self.device)

                factor=64
                H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
                padh = H-h if h%factor!=0 else 0
                padw = W-w if w%factor!=0 else 0
                noise = F.pad(noise, (0,padw,0,padh), 'reflect')
                clean = F.pad(clean, (0,padw,0,padh), 'reflect')

                denoise = torch.zeros_like(noise).to('cpu')

                denoise_list=[]

                for i in range(eval_time):
                    x = torch.randn(b, c, h, w, device=self.device)
                    denoise += self.sample_image(x=data_transform(noise),
                                                 eta=eta,
                                                 difussion_times=difussion_times,
                                                 )

                denoise = data_transform_reverse(denoise / eval_time).to(self.device)
                noise = noise[:,:,:h,:w]
                clean = clean[:,:,:h,:w]
                denoise = denoise[:,:,:h,:w]


                for i in range(len(denoise)):
                    if 'FMDD' in self.config['data']['test_path_clean']:
                        # current_lpips_distance = loss_fn.forward(denoise[i, :, :, :], clean[i, :, :, :])
                        # avg_lpips+=current_lpips_distance

                        clean_one = clean[i,0, :, :].cpu().detach().numpy()*255
                        clean_one = np.uint8((clean_one).round())
                        util.imsave(clean_one, self.config['save']['photo_path'] + str(idx) + '_clean.png')
                        util.imsave(clean_one, self.config['save']['photo_path'] +'clean/'+ str(idx) + '_clean.png')       
                        
                        denoise_one = denoise[i, :, :, :]
                        denoise_one = torch.clamp(denoise_one,0,1).cpu().detach().permute(1, 2, 0).numpy()
                        denoise_one = (denoise_one[:,:,0]+denoise_one[:,:,1]+denoise_one[:,:,2])/3*255
                        denoise_one = np.uint8((denoise_one).round())
                        util.imsave(denoise_one, self.config['save']['photo_path'] + str(idx) + '_denoise.png')
                        util.imsave(denoise_one, self.config['save']['photo_path'] +'denoise/'+ str(idx) + '_denoise.png')

                        noise_one = noise[i, :, :, :]
                        noise_one = util.tensor2uint(noise_one)

                        util.imsave(noise_one, self.config['save']['photo_path'] + str(idx) + '_noise.png')

                        idx = idx + 1
                        # print(denoise_one.shape,clean_one.shape)
                        current_psnr = util.calculate_psnr(denoise_one, clean_one)
                        avg_rgb_psnr += current_psnr
                        # print(current_psnr)

                        
                    else: 

                        # current_lpips_distance = loss_fn.forward(denoise[i, :, :, :], clean[i, :, :, :])
                        # avg_lpips+=current_lpips_distance

                        denoise_one = denoise[i, :, :, :]
                        denoise_one = util.tensor2uint(denoise_one)

                        util.imsave(denoise_one, self.config['save']['photo_path'] + str(idx) + '_denoise.png')
                        util.imsave(denoise_one, self.config['save']['photo_path'] +'denoise/'+ str(idx) + '_denoise.png')

                        clean_one = util.tensor2uint(clean[i, :, :, :])
                        util.imsave(clean_one, self.config['save']['photo_path'] + str(idx) + '_clean.png')
                        util.imsave(clean_one, self.config['save']['photo_path'] +'clean/'+ str(idx) + '_clean.png')
                        

                        noise_one = noise[i, :, :, :]
                        noise_one = util.tensor2uint(noise_one)

                        util.imsave(noise_one, self.config['save']['photo_path'] + str(idx) + '_noise.png')

                        idx = idx + 1

                        current_psnr = util.calculate_psnr(denoise_one, clean_one)
                        avg_rgb_psnr += current_psnr

        avg_rgb_psnr = avg_rgb_psnr / idx
        print('PSNR: ', avg_rgb_psnr)
        # print('LPIPS: ',avg_lpips/idx)
        print('SSIM: ',avg_rgb_ssim/idx)

        
        return avg_rgb_psnr #,(avg_lpips/idx).item() #,0#,metrics_dict['frechet_inception_distance']


if __name__ == "__main__":
    option = {
    'data': {
        'test_n_channels': 3,
        'test_H_size': 256,                                 #fixed,imagesize for ImageNet
        'test_path_noise':'',
        'test_path_clean':'',
        'test_batch_size': 1,
        'test_num_workers': 1,                              
        'test_opt': 'Kodak',                                #Dataset
        'noise_sigma':150                                   #test noise_lvel for Gaussian noise
    },

    'model': {
        'type': "openai",                                   #fixed
    },

    'diffusion': {
        'beta_schedule': 'linear',                          #fixed
        'beta_start': 0.0001,                               #fixed
        'beta_end': 0.02,                                   #fixed
        'num_diffusion_timesteps': 1000,                    #fixed
        'sampling_timesteps': 1                             #fixed
    },

    'save': {
        'photo_path': './results/photo_temp/',  
        'ddpm_checkpoint': './pre-trained/256x256_diffusion_uncond.pt'          #the path of the pre-trained diffusion model
    },

    'device_id': 'cuda',
    }
    
    import argparse
    parser = argparse.ArgumentParser(description='Real-world Image Denoising using DMID')
    parser.add_argument('--clean_path', default='./data/CC-full/GT/', type=str, help='for example: ./data/CC/clean/')
    parser.add_argument('--noisy_path', default='./data/CC-full/Noisy/', type=str, help='for example: ./data/CC/noisy/')
    parser.add_argument('--datatype', default='CC', type=str, help='CC/PolyU/FMDD')
    parser.add_argument('--S_t', default=1, type=int, help='sampling times in one inference')
    parser.add_argument('--R_t', default=1, type=int, help='repetition times of multiple inferences')
    args = parser.parse_args()

    datatype=args.datatype

    ###############################################################################
    #2：real-world denoising
    ###############################################################################
    option['data']['test_path_noise'] = args.noisy_path
    option['data']['test_path_clean'] = args.clean_path
    option['data']['test_opt']=  args.datatype
    S_t=args.S_t
    R_t=args.R_t
    os.makedirs('./results/'+'real-world/'+datatype+'/'+'/s'+str(S_t)           ,exist_ok=True)
    os.makedirs('./results/'+'real-world/'+datatype+'/'+'/s'+str(S_t)+'/clean'  ,exist_ok=True)
    os.makedirs('./results/'+'real-world/'+datatype+'/'+'/s'+str(S_t)+'/denoise',exist_ok=True)
    print('sample: '+str(S_t))
    option['diffusion']['sampling_timesteps']=S_t
    option['save']['photo_path']='./results/'+'real-world/'+datatype+'/'+'/s'+str(S_t)+'/'
    
    TRAIN=Train(config=option)
    TRAIN.eval_ddpm(eta=0.85,
                    eval_time=R_t,
                    difussion_times=0,
                    )
            


