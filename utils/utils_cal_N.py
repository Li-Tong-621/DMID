import torch
import numpy as np

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

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def precompute_ratio(betas):
    num_diffusion_timesteps = len(betas)
    t = torch.arange(num_diffusion_timesteps)
    a = compute_alpha(betas, t)
    ratio = torch.sqrt((1 - a) / a)
    return ratio

def find_best_t(ratio, sigma):
    abs_diff = torch.abs(ratio - sigma)
    best_t = torch.argmin(abs_diff).item()
    return best_t+1

def find_N(sigma=2*10/255):
    betas = get_beta_schedule(beta_schedule='linear',
                              beta_start=0.0001,
                              beta_end=0.02,
                              num_diffusion_timesteps=1000)
    betas = torch.from_numpy(betas).float()
    ratio = precompute_ratio(betas)
    # print(ratio)
    best_t = find_best_t(ratio, sigma)
    
    # for i in [15,25,50,100,150,200,250]:
    #     print(find_best_t(ratio, i*2/255))
    return best_t

print(find_N(sigma=2*200/255))