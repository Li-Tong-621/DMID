PolyU_dict={
    'LR' : 1e-2,
    'sigma' : 3,
    'rho' : 1,
    'eta' : 0.5,
    'total_step' : 50,
    'prob1_iter' : 500,
    'eps':0.04,
    'noise_level':[5 for i in range(100)]
}

CC_dict={
    'LR' : 1e-2,
    'sigma' : 3,
    'rho' : 1,
    'eta' : 0.5,
    'total_step' : 50,
    'prob1_iter' : 500,
    'eps':0.0003,
    'noise_level':[10 for i in range(15)]
}

FMDD_dict={
    'LR' : 1e-2,
    'sigma' : 5,
    'rho' : 1,
    'eta' : 0.5,
    'total_step' : 50,
    'prob1_iter' : 500,
    'eps':0.00001,
    'noise_level':[ 15,10,15,15,                  #Confocl_BPAE_B
                    10,10,10,15,                  #Confocl_BPAE_G 
                    10,10,5,10,                   #Confocl_BPAE_R
                    35,35,35,35,                  #Confocl_fish 
                    15,15,30,15,                  #Confocl_MICE 
                    25,20,40,40,                  #Twophoto_BPAE_B
                    20,20,20,20,                  #Twophoto_BPAE_G
                    10,5,10,10,                   #Twophoto_BPAE_R
                    20,20,20,20,                  #Twophoto_MICE
                    60,60,60,60,                  #Widefield_BPAE_B
                    50,50,50,50,                  #Widefield_BPAE_G
                    40,40,40,40]                  #Widefield_BPAE_R
}



