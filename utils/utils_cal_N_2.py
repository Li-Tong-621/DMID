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

def generalized_steps(n, seq, b, eta=0., sigma=15 / 255):
    for i in reversed(seq):
        t = (torch.ones(n) * i)
        at = compute_alpha(b, t.long())
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
    
    photo_noise = a.sqrt() * sigma
    
    t = (torch.ones(n) * difussion_times - 1)
    at = compute_alpha(betas, t.long())
    return photo_noise, (1 - at).sqrt()

def find_N(sigma=2*10/255):
    best_parameter=[]
    temp={'ans':2,
          'd_step':2,
          's_step':2}

    best_parameter.append(temp)
    for d_step in range(2,500):
        # for s_step in range(1,d_step+1):
        for s_step in [1]:
            
            photo_noise,model_noise=sample_image(n=1,
                                     eta=0.85,
                                     difussion_times=d_step,
                                     sigma=sigma,
                                     sampling_timesteps=s_step)
            
            if abs((model_noise/photo_noise -1).item()) < best_parameter[-1]['ans']:
                best_parameter.append({'ans':abs((model_noise/photo_noise -1).item()),
                                       'photo_noise':photo_noise.item(),
                                       'model_noise':model_noise.item(),
                                       'd_step':d_step,
                                       's_step':s_step})
            
    return best_parameter[-1]['d_step']

print(find_N(sigma=2*200/255))

