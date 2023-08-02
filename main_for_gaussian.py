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
import utils_image as util
from Data import Dataset
import utils_sampling
from collections import OrderedDict
import os
import torch.nn.functional as F
import lpips
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch_fidelity
loss_fn =lpips.LPIPS(net='alex', version='0.1').to('cuda')

def data_transform(X):
    return 2 * X - 1.0


def data_transform_reverse(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


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


class Train(object):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device_id = config['device_id']
        self.device = torch.device(self.device_id)


        self.model = UNetModel()
        #print(self.model)
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

        dataset_test = Dataset(n_channels=config['data']['test_n_channels'],
                               H_size=config['data']['test_H_size'],
                               path_noise=config['data']['test_path_noise'],
                               path_clean=config['data']['test_path_clean'],
                               opt=config['data']['test_opt'],
                               noise_sigma=config['data']['noise_sigma']
                               )

        self.val_loader = torch.utils.data.DataLoader(dataset_test,
                                                      batch_size=config['data']['test_batch_size'],
                                                      num_workers=config['data']['test_num_workers']
                                                      , pin_memory=True

                                                      )
        print('____________________________prepare_over____________________________')

    def sample_image(self,
                     x_cond,
                     x,
                     last=True,
                     eta=0.8,
                     difussion_times=500,
                     sample_type='DDIM',
                     travel_length=20, 
                     travel_repeat=3
                     ):
        #______________________embedding for gaussian___________________

        e = torch.randn_like(x_cond[:, :, :, :])
        t = torch.ones(x_cond.shape[0]).to(self.device)
        t = t*(difussion_times-1)
        t = torch.tensor(t,dtype=torch.int)
        a = (1 - self.betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        x_cond = x_cond[:, :, :, :] * a.sqrt()
        #____________________________  end  ____________________________

        
        skip = difussion_times // self.config['diffusion']['sampling_timesteps']
        seq = range(0, difussion_times, skip)
        seq=list(seq[:-1])+[difussion_times-1]  #a liilte different from ddim


        xs = utils_sampling.generalized_steps(x=x,
                                                x_cond=x_cond,
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
                  sample_type='DDIM',
                  travel_length=20, 
                  travel_repeat=3
                  ):

        avg_rgb_psnr = 0.0
        avg_lpips=0.0
        psnr_list = []

        idx = 0
        with torch.no_grad():
            for index, (noise, clean) in enumerate(self.val_loader, 0):
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

                denoise = torch.zeros_like(clean).to('cpu')

                denoise_list=[]

                for i in range(eval_time):
                    x = torch.randn(b, c, h, w, device=self.device)
                    denoise += self.sample_image(x_cond=data_transform(noise),
                                                 x=x,
                                                 eta=eta,
                                                 difussion_times=difussion_times,
                                                 sample_type=sample_type,
                                                 travel_length=travel_length,
                                                 travel_repeat=travel_repeat
                                                 )

                denoise = data_transform_reverse(denoise / eval_time).to(self.device)

                print('cost_time: ', time.time() - data_start)

                noise = noise[:,:,:h,:w]
                clean = clean[:,:,:h,:w]
                denoise = denoise[:,:,:h,:w]


                for i in range(len(denoise)):

                    
                    current_lpips_distance = loss_fn.forward(denoise[i, :, :, :], clean[i, :, :, :])
                    avg_lpips+=current_lpips_distance

                    denoise_one = denoise[i, :, :, :]
                    denoise_one = util.tensor2uint(denoise_one)

                    util.imsave(denoise_one, self.config['save']['photo_path'] + str(idx) + '_denoise.png')

                    clean_one = util.tensor2uint(clean[i, :, :, :])
                    util.imsave(clean_one, self.config['save']['photo_path'] + str(idx) + '_clean.png')
                    

                    noise_one = noise[i, :, :, :]
                    noise_one = util.tensor2uint(noise_one)

                    util.imsave(noise_one, self.config['save']['photo_path'] + str(idx) + '_noise.png')

                    idx = idx + 1

                    current_psnr = util.calculate_psnr(denoise_one, clean_one)
                    avg_rgb_psnr += current_psnr

        avg_rgb_psnr = avg_rgb_psnr / idx
        print('PSNR: ', avg_rgb_psnr)
        print('Lpips: ', avg_lpips/idx)

        return avg_rgb_psnr



option = {
    'data': {
        'test_n_channels': 3,
        'test_H_size': 256,
        'test_path_noise':'',
        'test_path_clean':'',
        'test_batch_size': 1,
        'test_num_workers': 1,
        'image_size': 256,
        'test_opt': 'Kodak-list',  #ImageNet / others,
        'noise_sigma':150
    },

    'model': {
        'type': "openai",
    },

    'diffusion': {
        'beta_schedule': 'linear',
        'beta_start': 0.0001,
        'beta_end': 0.02,
        'num_diffusion_timesteps': 1000,
        'sampling_timesteps': 1
    },

    'training': {
        'n_epochs': 200000,
    },

    'save': {
        'photo_path': './results/photo_temp/',
        'ddpm_checkpoint': './pre-trained/256x256_diffusion_uncond.pt',
    },

    'device_id': 'cuda',
}
import argparse
parser = argparse.ArgumentParser(description='Gasussian Color Denoising using DMID')

parser.add_argument('--data_path', default='./data/Imagenet', type=str, help='for example: ./data/Imagenet')
parser.add_argument('--dataset', default='ImageNet', type=str, help='ImageNet/CBSD68/Kodak24/McMaster/...')
parser.add_argument('--test_sigma', default=50, type=int, help='50/100/150/200/250/...')
parser.add_argument('--S_t', default=1, type=int, help='sampling times in one inference')
parser.add_argument('--R_t', default=1, type=int, help='inference times')
args = parser.parse_args()

d_t_list=[115,215,291,347,393]
s_t_list=[2,2,2,2,2]
noise_sigma_list=[50,100,150,200,250]

option['data']['test_path_noise'] = args.data_path
option['data']['test_path_clean'] = args.data_path
option['data']['test_opt'] = args.dataset
option['data']['noise_sigma']=args.test_sigma
option['diffusion']['sampling_timesteps']=args.S_t
N=d_t_list[ noise_sigma_list.index(args.test_sigma) ]
os.makedirs('./results/'+option['data']['test_opt'],exist_ok=True)
option['save']['photo_path']='./results/'+option['data']['test_opt']+'/'
print(option)
TRAIN=Train(config=option)
TRAIN.eval_ddpm(eta=0.85,
                eval_time=args.R_t,
                difussion_times=N,
                sample_type='DDIM'
                )





        






