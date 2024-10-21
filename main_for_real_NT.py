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
from utils.utils_option import *
#____________________________________________________________________________________________________
#____________________________________________diffusion model________________________________________
#____________________________________________________________________________________________________
import utils.utils_sampling as utils_sampling
import utils.utils_image as util
def data_transform(X):
    return 2 * X - 1.0

def data_transform_reverse(X):
    return (X + 1.0) / 2.0

def generalized_steps(n, seq, b, eta=0., sigma=15 / 255):
    for i in reversed(seq):
        t = (torch.ones(n) * i)
        at = utils_sampling.compute_alpha(b, t.long())
        return (1 - at).sqrt()

def sample_image(n,
                 eta=0.85,
                 difussion_times=500,
                 sigma=15 / 255,
                 sampling_timesteps=500,
                 ):
    # ____________________________renosie____________________________
    betas = get_beta_schedule(beta_schedule='linear',
                              beta_start=0.0001,
                              beta_end=0.02,
                              num_diffusion_timesteps=1000, )
    betas = torch.from_numpy(betas).float()
    # betas=torch.as_tensor(betas,dtype=torch.float64)
    t = torch.ones(n)
    t = t * (difussion_times - 1)
    t = torch.tensor(t, dtype=torch.int)
    a = (1 - betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    # print('diffusion_steps(N): ',difussion_times,' noise: ',a.sqrt()*sigma)
    photo_noise = a.sqrt() * sigma
    # ____________________________  end  ____________________________
    skip = difussion_times // sampling_timesteps
    seq = range(0, difussion_times, skip)
    seq = list(seq[:-1]) + [difussion_times - 1]
    model_noise = generalized_steps(n=n,
                                    seq=seq,
                                    b=betas,
                                    eta=eta,
                                    sigma=sigma)
    return photo_noise, model_noise


def find_N(sigma=2 * 15 / 255, range_low=1, range_high=250):
    best_parameter = []
    temp = {'ans': 2,
            'd_step': 13,
            's_step': 1}
    best_parameter.append(temp)
    for d_step in range(range_low, range_high):
        for s_step in [1]:
            photo_noise, model_noise = sample_image(n=1,
                                                    eta=0.85,
                                                    difussion_times=d_step,
                                                    sigma=sigma,
                                                    sampling_timesteps=s_step)
            if abs((model_noise / photo_noise - 1).item()) <= best_parameter[-1]['ans']:
                best_parameter.append({'ans': abs((model_noise / photo_noise - 1).item()),
                                       'photo_noise': photo_noise.item(),
                                       'model_noise': model_noise.item(),
                                       'd_step': d_step,
                                       's_step': s_step})
    for i in reversed(best_parameter):
        if i['s_step'] == 1:
            return i['d_step']
        
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
            difussion_times=find_N(2*noise_level/255)
            
            if 'FMDD' in datatype:
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


#____________________________________________________________________________________________________
#_______________________________________Noise Transformation________________________________________
#____________________________________________________________________________________________________

def sample_z(mean): 
    eps = mean.clone().normal_()
    
    return mean + eps
    
def eval_sigma(num_iter, noise_level):
    if num_iter>1 and noise_level>=30:
        return noise_level/2
    else:
        return noise_level   
        
def denoising(noise_im, 
              clean_im, 
              LR=1e-2, 
              sigma=5, 
              rho=1, 
              eta=0.5, 
              total_step=20, 
              prob1_iter=500, 
              result_root=None, 
              f=None,
              model=None,
              datatype='fmdd',
              eps=0.0001,
              noise_level=10):
    
    input_depth = 3
    latent_dim = 3  
    #________________________________________________
    if datatype=='FMDD':
        input_depth = 1
        latent_dim = 1
    #______________________________________________
    
    en_net = UNet(input_depth, latent_dim).to(device)
    de_net = UNet(latent_dim, input_depth).to(device)

    parameters = [p for p in en_net.parameters()] + [p for p in de_net.parameters()]         
    optimizer = torch.optim.Adam(parameters, lr=LR)    
    l2_loss = torch.nn.MSELoss().cuda()

    
    i0 = np_to_torch(noise_im).to(device)
    noise_im_torch = np_to_torch(noise_im).to(device)
    i0_til_torch = np_to_torch(noise_im).to(device)
    Y = torch.zeros_like(noise_im_torch).to(device)
    
    
    best_sure = 99999999999999999
    for i in range(total_step):
        for i_1 in range(prob1_iter):
            
            optimizer.zero_grad()

            mean = en_net(noise_im_torch)
            z = sample_z(mean)
            out = de_net(z)
            
            total_loss =  0.5 * l2_loss(out, noise_im_torch)
            total_loss += 0.5 * (1/sigma**2)*l2_loss(mean, i0)
            total_loss += (rho/2) * l2_loss(i0 + Y, i0_til_torch)
            
            total_loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                i0 = ((1/sigma**2)*mean.detach() + rho*(i0_til_torch - Y)) / ((1/sigma**2) + rho)
            
        
        with torch.no_grad():
            temp=i0+Y
            i0_np = torch_to_np(i0)
            Y_np = torch_to_np(Y)
            
            sig = eval_sigma(i+1, noise_level)
            
            if datatype=='FMDD':
                i0_til_np = bm3d.bm3d((i0_np + Y_np)/255, sig/255)*255
                i0_til_torch = np_to_torch(i0_til_np).to(device)
            else:
                i0_til_np = bm3d.bm3d_rgb(i0_np.transpose(1, 2, 0) + Y_np.transpose(1, 2, 0), sig).transpose(2, 0, 1)
                i0_til_torch = np_to_torch(i0_til_np).to(device)
            
            # i0_til_torch = np_to_torch(i0_til_np).to(device).to(torch.float32)
            
            Y = Y + eta * (i0 - i0_til_torch)
            
            # Y_np = torch_to_np(Y)

            if datatype=='FMDD':
                n=torch.randn(temp.size()).cuda()
                batch=1
                var=(sig/255.0)**2
                x_dim=noise_im.shape[1]
                Z_ = bm3d.bm3d((i0_np + Y_np+eps*torch_to_np(n))/255, (sig+eps)/255)*255
                Z_ = np_to_torch(Z_).to(device)
                divergence = (1.0/eps)*(torch.sum(torch.mul(n, (Z_-i0_til_torch))))
                sure = (1.0 / batch)*(2.0*l2_loss(temp , i0_til_torch) - batch*x_dim*x_dim*var + 2.0*var*divergence)
                
            else:
                n=torch.randn(temp.size()).cuda()
                batch=1
                var=(sig/255.0)**2
                x_dim=noise_im.shape[1]
                Z_ = bm3d.bm3d_rgb(i0_np.transpose(1, 2, 0) + Y_np.transpose(1, 2, 0)+eps*torch_to_np(n).transpose(1, 2, 0), sig+eps).transpose(2, 0, 1)
                Z_ = np_to_torch(Z_).to(device)
                divergence = (1.0/eps)*(torch.sum(torch.mul(n, (Z_-i0_til_torch))))
                sure = (1.0 / batch)*(2.0*l2_loss(temp , i0_til_torch) - batch*x_dim*x_dim*var + 2.0*var*divergence)
                
            
        
            from skimage.metrics import structural_similarity as compare_ssim
            from skimage.metrics import peak_signal_noise_ratio as compare_psnr
            
            i0_til_np = torch_to_np(i0_til_torch).clip(0, 255)
            
            i0_til_pil = np_to_pil(i0_til_np)
            # i0_til_pil.save(os.path.join(result_root, '{}'.format(i) + '.png'))
            # print(f)
            print('Iteration: {:02d}, VAE Loss: {:f}, sure: {:f}'.format(i, 
                        total_loss.item(), 
                        sure), 
                        file=f, flush=True
                  )
            if best_sure > sure:
                best_sure = sure
                Noisy_optimal=temp.clone()
                Noise_level=sig
            else:
                break
            
    return i0_til_np, Noise_level, Noisy_optimal



import argparse
parser = argparse.ArgumentParser(description='Real-world Image Denoising using DMID')
parser.add_argument('--clean_path', default='./data/CC/GT/', type=str, help='for example: ./data/CC/clean/')
parser.add_argument('--noisy_path', default='./data/CC/Noisy/', type=str, help='for example: ./data/CC/noisy/')
parser.add_argument('--datatype', default='CC', type=str, help='CC/PolyU/FMDD')
parser.add_argument('--pertrianed', default='None', type=str, help='None for getting the embedding by yourself;others for directly searching for better results')
parser.add_argument('--S_t', default=1, type=int, help='sampling times in one inference')
parser.add_argument('--R_t', default=1, type=int, help='repetition times of multiple inferences')
args = parser.parse_args()

datatype=args.datatype
###############################################################################
#1.noise transformation
###############################################################################

if args.pertrianed=='None':
    if datatype=='CC':
        noises = natsorted(glob.glob(args.noisy_path + '*.png'))
        cleans = natsorted(glob.glob(args.clean_path + '*.png'))
        name=-9
        data_dct=CC_dict
    elif datatype=='PolyU':
        noises = natsorted(glob.glob(args.noisy_path + '*.JPG'))
        cleans = natsorted(glob.glob(args.clean_path + '*.JPG'))   
        name=-9
        data_dct=PolyU_dict
    elif datatype=='FMDD':
        noises = natsorted(glob.glob(args.noisy_path + '*.png'))
        cleans = natsorted(glob.glob(args.clean_path + '*.png'))     
        name=-4
        data_dct=FMDD_dict
    
    else:
        #####other datasets please moified by yourself
        noises = natsorted(glob.glob(args.noisy_path + '*.png'))
        cleans = natsorted(glob.glob(args.clean_path + '*.png'))
        name= -4
        data_dct=CC_dict
        
    LR = data_dct['LR']
    rho = data_dct['rho']
    eta = data_dct['eta']
    total_step = data_dct['total_step']
    prob1_iter = data_dct['prob1_iter']
    sigma=data_dct['sigma']
    
    save_hidden={}
    save_idx=0
    for noise, clean in zip(noises, cleans):
        result = 'results/nn_'+str(datatype)+'/{}/'.format(noise.split('/')[-1][:name])
        os.system('mkdir -p ' + result)
        
        
        noise_im = Image.open(noise)
        clean_im = Image.open(clean)
        noise_im_np = pil_to_np(noise_im)
        clean_im_np = pil_to_np(clean_im)
    
        #a higher noise usually corresponds to a lower eps 
        eps=data_dct['eps']
        noise_level=data_dct['noise_level'][save_idx]
        
        with open(result + 'result.txt', 'w') as f:
            _, noise_level, Noisy_optimal= denoising(noise_im_np, clean_im_np, LR=LR, sigma=sigma, rho=rho, eta=eta, 
                                        total_step=total_step, prob1_iter=prob1_iter, result_root=result, f=f,model=model,
                                        datatype=datatype,eps=eps,noise_level=noise_level)
            
        save_hidden[str(save_idx)]={'noise_level':copy.copy(noise_level),
                                    'noise_image':Noisy_optimal.clone()}
        torch.save(save_hidden, './results/nn_'+str(datatype)+'/'+str(datatype)+'.pt')
        save_idx+=1 
    print('noise transformation has finished!')

# ###############################################################################
# #2：real-world denoising. We recomand to run main_for_real.py, as the following denising code may exist errors.
# ###############################################################################
# s_t=args.S_t
# R_t=args.R_t
# os.makedirs('./results/'+'real-world/'+datatype+'/'+'/s'+str(s_t)           ,exist_ok=True)
# os.makedirs('./results/'+'real-world/'+datatype+'/'+'/s'+str(s_t)+'/clean'  ,exist_ok=True)
# os.makedirs('./results/'+'real-world/'+datatype+'/'+'/s'+str(s_t)+'/denoise',exist_ok=True)
# print('sample: '+str(s_t))
# eval_ddpm(eta=0.85,
#             eval_time=R_t,
#             difussion_times=0,
#             sample_times=s_t,
#             datatype=datatype,
#             cleanpath=args.clean_path,
#             savepth='./results/'+'real-world/'+datatype+'/'+'/s'+str(s_t)+'/',
#             latent_path=args.pertrianed
#             )
