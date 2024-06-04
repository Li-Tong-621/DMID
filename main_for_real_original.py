from utils.utils_nn import Encoder, Decoder
import torch
from utils.utils_nn import pil_to_np, np_to_pil, np_to_torch, torch_to_np,UNet
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import copy
import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
device=torch.device('cuda')    
from natsort import natsorted
import torchvision.transforms as transforms
import PIL
import time
import torch.nn.functional as F
import bm3d

#____________________________________________________________________________________________________
#____________________________________________diffusion model________________________________________
#____________________________________________________________________________________________________
import utils.utils_sampling as utils_sampling
import utils.utils_image as util
def data_transform(X):
    return 2 * X - 1.0

def data_transform_reverse(X):
    return (X + 1.0) / 2.0

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


betas = get_beta_schedule(
            beta_schedule="linear",
            beta_start=0.0001,
            beta_end=0.02,
            num_diffusion_timesteps=1000, )

betas = torch.from_numpy(betas).float().to(device)
from guided_diffusion.unet import UNetModel
model = UNetModel()
model.to(device)
model.load_state_dict(torch.load('./pre-trained/256x256_diffusion_uncond.pt', map_location=device))
model.eval()
def sample_image_d(x,
                last=True,
                eta=0.8,
                difussion_times=500,
                sample_times=1,
                ):
    #______________________convert to x_N___________________
    y = x.clone()
    t = torch.ones(x.shape[0]).to(device)
    t = t*(difussion_times-1)
    t = torch.tensor(t,dtype=torch.int)
    a = (1 - betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x_N = x[:, :, :, :] * a.sqrt()
    #______embedding for real-world denoising concludes_____
    # print(difussion_times,sample_times)
    skip = difussion_times // sample_times
    if sample_times>difussion_times:
        skip=1
    seq = range(0, difussion_times, skip)
    if sample_times==1:
        seq = [difussion_times-1]

    xs = utils_sampling.generalized_steps(
                                            x=x_N,
                                            seq=seq,
                                            model=model,
                                            b=betas,
                                            eta=eta)
    return xs[0][-1]


def eval_ddpm(eta=0.85,
                eval_time=1,
                difussion_times=500,
                sample_times=2,
                datatype='fmdd',
                cleanpath='',
                savepth='',
                latent_path=''
                ):
    
    idx = 0

    if datatype!='PolyU':
        cleans = natsorted(glob.glob(cleanpath + '*.png'))
    else:
        cleans = natsorted(glob.glob(cleanpath + '*.JPG'))
    # print(cleans)
    
    if latent_path==None:
        noises = torch.load('./results/nn_'+str(datatype)+'/'+str(datatype)+'.pt', map_location='cpu')
    else:
        noises = torch.load(latent_path, map_location='cpu')
    trans=transforms.ToTensor()

    with torch.no_grad():
        for index in range(len(cleans)):
            
            noise=noises[str(index)]['noise_image']/255
            noise_level=noises[str(index)]['noise_level']
            if 'CC' in datatype:
                difussion_times=21

            if 'PolyU' in datatype:
                difussion_times=9
            
            if 'FMDD' in datatype:
                difussion_times=[35,21,33,33,
                    16,21,16,33,
                    18,18,9,21,
                    39,39,81,39,
                    38,33,69,33,
                    57,45,93,45,
                    45,45,45,45,
                    21,9,21,21,
                    45,45,45,45,
                    137,137,137,137,
                    115,115,115,115,
                    93,93,45,93
                ][index]
                noise=noise.repeat(1,3,1,1)
                    
            clean=PIL.Image.open(cleans[index])
            clean=trans(clean)
            clean=torch.unsqueeze(clean, 0)

            data_start = time.time()
            b, c, h, w = noise.shape
            clean = clean.to(device)
            noise = noise.to(device)

            factor=64
            H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            noise = F.pad(noise, (0,padw,0,padh), 'reflect')
            
            denoise = torch.zeros_like(noise).to('cpu')

            for i in range(eval_time):
                denoise += sample_image_d(x=data_transform(noise),
                                        eta=eta,
                                        difussion_times=difussion_times,
                                        sample_times=sample_times)

            denoise = data_transform_reverse(denoise / eval_time).to(device)
            print('cost_time: ', time.time() - data_start)

            noise = noise[:,:,:h,:w]
            denoise = denoise[:,:,:h,:w]


            for i in range(len(denoise)):

                
                if datatype!='FMDD':
                    denoise_one = denoise[i, :, :, :]
                    denoise_one = util.tensor2uint(denoise_one)
                    clean_one = util.tensor2uint(clean[i, :, :, :])
                else:
                    denoise_one = denoise[i, :, :, :]
                    denoise_one = torch.clamp(denoise_one,0,1).cpu().detach().permute(1, 2, 0).numpy()
                    denoise_one = (denoise_one[:,:,0]+denoise_one[:,:,1]+denoise_one[:,:,2])/3*255
                    denoise_one = np.uint8((denoise_one).round())

                    clean_one = clean[i,0, :, :].cpu().detach().numpy()*255
                    clean_one = np.uint8((clean_one).round())

                util.imsave(denoise_one, savepth + str(idx) + '_denoise.png')
                util.imsave(denoise_one, savepth +'denoise/'+ str(idx) + '_denoise.png')
                
                util.imsave(clean_one, savepth + str(idx) + '_clean.png')
                util.imsave(clean_one, savepth +'clean/'+ str(idx) + '_clean.png')
        
                noise_one = noise[i, :, :, :]
                noise_one = util.tensor2uint(noise_one)
                util.imsave(noise_one, savepth + str(idx) + '_noise.png')

                idx = idx + 1

    print('Finish!')

import argparse
parser = argparse.ArgumentParser(description='Real-world Image Denoising using DMID')
parser.add_argument('--clean_path', default='./data/CC-full/GT/', type=str, help='for example: ./data/CC/clean/')
parser.add_argument('--noisy_path', default='./data/CC-full/Noisy/', type=str, help='for example: ./data/CC/noisy/')
parser.add_argument('--datatype', default='CC', type=str, help='CC/PolyU/FMDD')
parser.add_argument('--pertrianed', default='./pre-trained/CC.pt', type=str, help='None for getting the embedding by yourself;others for directly searching for better results')
parser.add_argument('--S_t', default=1, type=int, help='sampling times in one inference')
parser.add_argument('--R_t', default=1, type=int, help='repetition times of multiple inferences')
args = parser.parse_args()

datatype=args.datatype

###############################################################################
#2：real-world denoising
###############################################################################
s_t=args.S_t
R_t=args.R_t
os.makedirs('./results/'+'real-world/'+datatype+'/'+'/s'+str(s_t)           ,exist_ok=True)
os.makedirs('./results/'+'real-world/'+datatype+'/'+'/s'+str(s_t)+'/clean'  ,exist_ok=True)
os.makedirs('./results/'+'real-world/'+datatype+'/'+'/s'+str(s_t)+'/denoise',exist_ok=True)
print('sample: '+str(s_t))
eval_ddpm(eta=0.85,
            eval_time=R_t,
            difussion_times=0,
            sample_times=s_t,
            datatype=datatype,
            cleanpath=args.clean_path,
            savepth='./results/'+'real-world/'+datatype+'/'+'/s'+str(s_t)+'/',
            latent_path=args.pertrianed
            )
