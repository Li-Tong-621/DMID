from nn_util import Encoder, Decoder
import torch
from nn_util import pil_to_np, np_to_pil, np_to_torch, torch_to_np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import copy
import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device=torch.device('cuda')    
from natsort import natsorted
import lpips
import torch_fidelity
loss_fn =lpips.LPIPS(net='alex', version='0.1').to('cuda')
import torchvision.transforms as transforms
import PIL
import time
import torch.nn.functional as F
#____________________________________________________________________________________________________
#____________________________________________diffusion model________________________________________
#____________________________________________________________________________________________________
import utils_sampling
import utils_image as util
def data_transform(X):
    return 2 * X - 1.0


def data_transform_reverse(X):
    #return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)
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
def sample_image_d(x_cond,
                last=True,
                eta=0.8,
                difussion_times=500,
                sample_times=1,
                ):

    y = x_cond.clone()
    t = torch.ones(x_cond.shape[0]).to(device)
    t = t*(difussion_times-1)
    t = torch.tensor(t,dtype=torch.int)
    a = (1 - betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x_cond = x_cond[:, :, :, :] * a.sqrt()
    skip = difussion_times // sample_times
    seq = range(0, difussion_times, skip)
    if sample_times==1:
        seq = [difussion_times-1]

    xs = utils_sampling.generalized_steps(x=x_cond,
                                            x_cond=x_cond,
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
    avg_lpips=0.0
    avg_rgb_psnr = 0.0
    idx = 0


    if datatype!='polyu':
        cleans = natsorted(glob.glob(cleanpath + '*.png'))
    else:
        cleans = natsorted(glob.glob(cleanpath + '*.JPG'))
    #print(cleans)
    if latent_path=='no':
        noises = torch.load('./results/nn_'+str(datatype)+'/'+str(datatype)+'.pt', map_location='cpu')
    else:
        noises = torch.load(latent_path, map_location='cpu')
    trans=transforms.ToTensor()

    with torch.no_grad():
        for index in range(len(cleans)):
            
            noise=noises[str(index)]['noise_image']
            difussion_times=noises[str(index)]['N']
            clean=PIL.Image.open(cleans[index])
            clean=trans(clean)
            clean=torch.unsqueeze(clean, 0)

            data_start = time.time()
            b, c, h, w = noise.shape

            clean = clean.to(device)
            noise = noise.to(device)

            #print(clean.shape,noise.shape)

            factor=64
            H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            noise = F.pad(noise, (0,padw,0,padh), 'reflect')
            # clean = F.pad(clean, (0,padw,0,padh), 'reflect')

            denoise = torch.zeros_like(noise).to('cpu')

            denoise_list=[]

            for i in range(eval_time):
                x = torch.randn(b, c, h, w, device=device)
                denoise += sample_image_d(x_cond=noise,
                                        eta=eta,
                                        difussion_times=difussion_times,
                                        sample_times=sample_times)

            denoise = data_transform_reverse(denoise / eval_time).to(device)

            print('cost_time: ', time.time() - data_start)

            noise = noise[:,:,:h,:w]
            denoise = denoise[:,:,:h,:w]


            for i in range(len(denoise)):

                # current_lpips_distance = loss_fn.forward(denoise[i, :, :, :], clean[i, :, :, :])
                # avg_lpips+=current_lpips_distance
                if datatype!='fmdd':
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
                #print(savepth +'denoise/'+ str(idx) + '_denoise.png')
                util.imsave(clean_one, savepth + str(idx) + '_clean.png')
                util.imsave(clean_one, savepth +'clean/'+ str(idx) + '_clean.png')
                
                current_lpips_distance = loss_fn(lpips.im2tensor(lpips.load_image(savepth +'clean/'+ str(idx) + '_clean.png')).to('cuda'),lpips.im2tensor(lpips.load_image(savepth +'denoise/'+ str(idx) + '_denoise.png')).to('cuda')).to('cpu').detach().item()
                avg_lpips+=current_lpips_distance
                
                noise_one = noise[i, :, :, :]
                noise_one = util.tensor2uint(noise_one)

                util.imsave(noise_one, savepth + str(idx) + '_noise.png')

                idx = idx + 1

                current_psnr = util.calculate_psnr(denoise_one, clean_one)
                avg_rgb_psnr += current_psnr



    avg_rgb_psnr = avg_rgb_psnr / idx
    print('PSNR: ', avg_rgb_psnr)
    print('LPIPS: ',avg_lpips/idx)

    metrics_dict = torch_fidelity.calculate_metrics(
    input1=savepth +'denoise/',
    input2=savepth +'clean/',
    cuda=True,
    isc=False,
    fid=True,
    kid=False,
    verbose=False,
    batch_size=1,
    save_cpu_ram=True, )
    print('FID',metrics_dict['frechet_inception_distance'])

    return avg_rgb_psnr,avg_lpips/idx,metrics_dict['frechet_inception_distance']



#____________________________________________________________________________________________________
#____________________________________________diffusion model________________________________________
#____________________________________________________________________________________________________
def save_hist(x, root):
    x = x.flatten()
    plt.figure()
    n, bins, patches = plt.hist(x, bins=128, density=1)
    plt.savefig(root)
    plt.close()
    
def save_heatmap(image_np, root):
    cmap = plt.get_cmap('jet')
    
    rgba_img = cmap(image_np)
    rgb_img = np.delete(rgba_img, 3, 2)
    rgb_img_pil = Image.fromarray((255*rgb_img).astype(np.uint8))
    rgb_img_pil.save(root)
    
def kl_loss(mean, log_var, mu, sigma):
    kl = 0.5*((1/(sigma**2))*(mean - mu)**2 + torch.exp(log_var)/(sigma**2) + np.log(sigma**2) - log_var - 1)
    loss = torch.mean(torch.sum(kl, dim=(1, 2, 3)))
    
    return loss

def sample_z(mean, log_var):
    eps = mean.clone().normal_()*torch.exp(log_var/2)
    
    return mean + eps    

def denoising(noise_im, clean_im, LR=1e-2, sigma=5, rho=1, eta=0.5, 
              total_step=20, prob1_iter=500, result_root=None, f=None,model=None,diffussion_min=2,diffusion_max=150,datatype='fmdd'):
    
    input_depth = 3
    latent_dim = 3
    
    en_net = Encoder(input_depth, latent_dim, down_sample_norm='batchnorm', 
                     up_sample_norm='batchnorm').cuda()
    de_net = Decoder(latent_dim, input_depth, down_sample_norm='batchnorm', 
                     up_sample_norm='batchnorm').cuda()
    
    #________________________________________________
    if datatype=='fmdd':
        original_clean=clean_im
        shape_c,h,w=np.shape(noise_im)
        
        noise_im=np.repeat(noise_im, 3, axis=0)
        clean_im=np.repeat(clean_im, 3, axis=0)
    #______________________________________________

    noise_im_torch = np_to_torch(noise_im)
    noise_im_torch = noise_im_torch.cuda()

    parameters = [p for p in en_net.parameters()] + [p for p in de_net.parameters()]             
    optimizer = torch.optim.Adam(parameters, lr=LR)    
    l2_loss = torch.nn.MSELoss(reduction='sum').cuda()
    
    i0 = np_to_torch(noise_im).cuda()
    Y = torch.zeros_like(noise_im_torch).cuda()
    i0_til_torch = np_to_torch(noise_im).cuda()

    diff_original_np = noise_im.astype(np.float32) - clean_im.astype(np.float32)
    diff_original_name = 'Original_dis.png'
    save_hist(diff_original_np, result_root+diff_original_name)  
    
    best_psnr = 0
    best_ssim = 0
    temp_psnr=0
    psnr=0
    for i in range(total_step):

############################### sub-problem 1 #################################
        for i_1 in range(prob1_iter):
            
            optimizer.zero_grad()
            
            mean, log_var = en_net(noise_im_torch)
        
            z = sample_z(mean, log_var)
            out = de_net(z)

            total_loss =  0.5 * l2_loss(out, noise_im_torch)
            total_loss += 0.5 * 1/(sigma**2) * l2_loss(mean, i0)
            
            total_loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                i0 = ((1/sigma**2)*mean + rho*(i0_til_torch - Y)) / ((1/sigma**2) + rho)

            
            if (i_1+1)%200==0:
                print(i_1+1)
                print("______________________________________")
                original_i0_til_torch=i0_til_torch.clone()
                with torch.no_grad():
                    
        ############################### sub-problem 2 #################################       
                    #print(torch.max(i0+Y),torch.mean(i0+Y))
                    temp=(i0+Y)
                    temp=data_transform(temp)

                    repeat_time=1
                    difussion_times=50
                    sample_times=1
                    
                    for difussion_times in range(diffussion_min,diffusion_max):
                        denoise = torch.zeros_like(temp).to('cpu')
                        for r_t in range(repeat_time):
                            #temp_results = torch.randn_like(temp, device=device)
                            denoise += sample_image_d(
                                                    x_cond=temp,
                                                    eta=0.85,
                                                    difussion_times=difussion_times,
                                                    sample_times=sample_times,
                                                    )
                        i0_til_torch=denoise/repeat_time
                        i0_til_torch=data_transform_reverse(i0_til_torch).cuda()
                        if datatype=='fmdd':
                            temp_noise_PSNR=(i0_til_torch[:,0,:,:]+i0_til_torch[:,1,:,:]+i0_til_torch[:,2,:,:])/3
                            temp_noise_PSNR=temp_noise_PSNR.data.float().clamp_(0, 1).cpu().numpy()
                            psnr = util.calculate_psnr(np.uint8((original_clean*255.0).round()), np.uint8((temp_noise_PSNR*255.0).round()))
                        else:
                            psnr = util.calculate_psnr(np.uint8((clean_im.transpose(1, 2, 0)*255.0).round()), util.tensor2uint(i0_til_torch))

                        
                        print('d ',difussion_times,'s ',sample_times,'psnr ',psnr)
                        if temp_psnr<psnr:
                            temp_psnr=psnr
                            temp_i0_til_torch=i0_til_torch.clone()
                            N_optimal=copy.copy(difussion_times)
                            Noisy_optimal=temp.clone()

                        
                    psnr=temp_psnr
                    print('psnr: ',psnr)
        
        
                i0_til_torch=original_i0_til_torch.clone()
############################### sub-problem 3 #################################           
        with torch.no_grad():
            Y = Y + eta * (i0 - i0_til_torch)
###############################################################################
            #正常按照之前的步骤优化，不过记录下来了最佳结果。
            i0_til_torch=temp_i0_til_torch.clone()
            i0_np = torch_to_np(i0)
            Y_np = torch_to_np(Y)
            denoise_obj_pil = np_to_pil((i0_np+Y_np).clip(0,1))
            Y_norm_np = np.sqrt((Y_np*Y_np).sum(0))
            i0_pil = np_to_pil(i0_np)
            mean_np = torch_to_np(mean)
            mean_pil = np_to_pil(mean_np)
            out_np = torch_to_np(out)
            out_pil = np_to_pil(out_np)
            diff_np = mean_np - clean_im
            
            denoise_obj_name = 'denoise_obj_{:04d}'.format(i) + '.png'
            Y_name = 'Y_{:04d}'.format(i) + '.png'
            i0_name = 'i0_num_epoch_{:04d}'.format(i) + '.png'
            mean_i_name = 'Latent_im_num_epoch_{:04d}'.format(i) + '.png'
            out_name = 'res_of_dec_num_epoch_{:04d}'.format(i) + '.png'
            diff_name = 'Latent_dis_num_epoch_{:04d}'.format(i) + '.png'
            
            denoise_obj_pil.save(result_root + denoise_obj_name)
            save_heatmap(Y_norm_np, result_root + Y_name)
            i0_pil.save(result_root + i0_name)
            mean_pil.save(result_root + mean_i_name)
            out_pil.save(result_root + out_name)
            save_hist(diff_np, result_root + diff_name)
            i0_til_np = torch_to_np(i0_til_torch).clip(0, 1)
            
            ssim = 0
            if datatype=='fmdd':
                #__________________________________________________________________
                i0_til_np=(i0_til_np[0,:,:]+i0_til_np[1,:,:]+i0_til_np[2,:,:])/3
                i0_til_np=np.expand_dims(i0_til_np,axis=0)
                i0_til_pil = np_to_pil(i0_til_np)
                i0_til_pil.save(os.path.join(result_root, '{}'.format(i) + '.png'))
                #______________________________________________________________________
            else:
                i0_til_pil = np_to_pil(i0_til_np)
                i0_til_pil.save(os.path.join(result_root, '{}'.format(i) + '.png'))

            print('Iteration: %02d, VAE Loss: %f, PSNR: %f, SSIM: %f' % (i, total_loss.item(), psnr, ssim), file=f, flush=True)
            print(psnr)
            if best_psnr < psnr:
                best_psnr = psnr
                best_ssim = ssim
            else:
                break
            i0_til_torch=original_i0_til_torch.clone()
            
    return i0_til_np, best_psnr, best_ssim, N_optimal, Noisy_optimal





















###############################################################################
#第一阶段：获得隐空间表达
###############################################################################
import argparse
parser = argparse.ArgumentParser(description='Real-world Image Denoising using DMID')
parser.add_argument('--clean_path', default='./data/CC/GT/', type=str, help='for example: ./data/CC/clean/')
parser.add_argument('--noisy_path', default='./data/CC/Noisy/', type=str, help='for example: ./data/CC/noisy/')
parser.add_argument('--datatype', default='cc', type=str, help='cc/polyu/fmdd')
# parser.add_argument('--pertrianed', default='./pre-trained/cc-latent.pt', type=str, help='no for getting the embedding by yourself;others for directly searching for better results')
parser.add_argument('--pertrianed', default='./results/nn_cc/cc.pt', type=str, help='no for getting the embedding by yourself;others for directly searching for better results')
parser.add_argument('--objective', default='DMID-p', type=str, help='objective,DMID-d/DMID-p/none')
args = parser.parse_args()

datatype=args.datatype

if args.pertrianed=='no':
    if datatype!='polyu':
        noises = natsorted(glob.glob(args.noisy_path + '*.png'))
        cleans = natsorted(glob.glob(args.clean_path + '*.png'))
    else:
        noises = natsorted(glob.glob(args.noisy_path + '*.JPG'))
        cleans = natsorted(glob.glob(args.clean_path + '*.JPG'))        

    if datatype=='fmdd':
        LR = 1e-2
        sigma = 0.1
        rho = 1
        eta = 0.5
        total_step = 20
        prob1_iter = 3001
    else:
        LR = 1e-2
        sigma = 0.5
        rho = 2
        eta = 0.5
        total_step = 20
        prob1_iter = 3001

    psnrs = []
    ssims = []
    save_hidden={}
    save_idx=0
    for noise, clean in zip(noises, cleans):
        result = 'results/nn_'+str(datatype)+'/{}/'.format(noise.split('/')[-1][:-9])
        os.system('mkdir -p ' + result)
        
        noise_im_pil = Image.open(noise)
        clean_im_pil = Image.open(clean)
        
        noise_im_np = pil_to_np(noise_im_pil)
        clean_im_np = pil_to_np(clean_im_pil)

        if datatype=='fmdd':
            if  'WideField' in noise:
                    diffussion_min=20
                    diffusion_max=150
            else:
                diffussion_min=2
                diffusion_max=70
        elif datatype=='cc':
            diffussion_min=2
            diffusion_max=50
        elif datatype=='polyu':
            diffussion_min=2
            diffusion_max=30

        with open(result + 'result.txt', 'w') as f:
            _, psnr, ssim, N_optimal, Noisy_optimal= denoising(noise_im_np, clean_im_np, LR=LR, sigma=sigma, rho=rho, eta=eta, 
                                        total_step=total_step, prob1_iter=prob1_iter, result_root=result, f=f,model=model,
                                        diffussion_min=diffussion_min,diffusion_max=diffusion_max,datatype=datatype)
            psnrs.append(psnr)
            ssims.append(ssim)
        save_hidden[str(save_idx)]={'N':copy.copy(N_optimal),
                                    'noise_image':Noisy_optimal.clone()}
        torch.save(save_hidden, './results/nn_'+str(datatype)+'/'+str(datatype)+'.pt')
        save_idx+=1 
    print('PSNR: {}'.format(sum(psnrs)/len(psnrs)))

###############################################################################
#第二阶段：寻优
###############################################################################

if args.objective=='DMID-d':
    s_t_list=[3] #for DMID-d
    R_t=333 #for DMID-d
elif args.objective=='DMID-p':
    s_t_list=[1,2,3] #for DMID-p
    R_t=1 #for DMID-p
if args.objective!='none':
    for s_t in s_t_list:
        for times in range(3):

            os.makedirs('./results/'+'real-world/'+datatype+'/'+'/s'+str(s_t)+'times_'+str(times)           ,exist_ok=True)
            os.makedirs('./results/'+'real-world/'+datatype+'/'+'/s'+str(s_t)+'times_'+str(times)+'/clean'  ,exist_ok=True)
            os.makedirs('./results/'+'real-world/'+datatype+'/'+'/s'+str(s_t)+'times_'+str(times)+'/denoise',exist_ok=True)
            print('sample: '+str(s_t))
            eval_ddpm(eta=0.85,
                        eval_time=R_t,
                        difussion_times=0,
                        sample_times=s_t,
                        datatype=datatype,
                        cleanpath=args.clean_path,
                        savepth='./results/'+'real-world/'+datatype+'/'+'/s'+str(s_t)+'times_'+str(times)+'/',
                        latent_path=args.pertrianed
                        )
