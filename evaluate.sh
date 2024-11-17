#If you want to reproduce quickly, you can reduce the repetitions times R_t. For example, setting R_ T=1 or R_ T=10, the approximate results can be obtained.

#TABLE 1, Classical Gaussian image denoising:

#CBSD68
python main_for_gaussian.py --data_path './data/CBSD68' --dataset 'CBSD68_15' --test_sigma 15 --S_t 2 --R_t 500
python main_for_gaussian.py --data_path './data/CBSD68' --dataset 'CBSD68_25' --test_sigma 25 --S_t 2 --R_t 500
python main_for_gaussian.py --data_path './data/CBSD68' --dataset 'CBSD68_50' --test_sigma 50 --S_t 2 --R_t 500
#Kodak24
python main_for_gaussian.py --data_path './data/Kodak24' --dataset 'Kodak24_15' --test_sigma 15 --S_t 2 --R_t 500
python main_for_gaussian.py --data_path './data/Kodak24' --dataset 'Kodak24_25' --test_sigma 25 --S_t 2 --R_t 500
python main_for_gaussian.py --data_path './data/Kodak24' --dataset 'Kodak24_50' --test_sigma 50 --S_t 2 --R_t 500
#McMaster
python main_for_gaussian.py --data_path './data/McMaster' --dataset 'McMaster_15' --test_sigma 15 --S_t 2 --R_t 500
python main_for_gaussian.py --data_path './data/McMaster' --dataset 'McMaster_25' --test_sigma 25 --S_t 2 --R_t 500
python main_for_gaussian.py --data_path './data/McMaster' --dataset 'McMaster_50' --test_sigma 50 --S_t 2 --R_t 500
#Urban100
python main_for_gaussian.py --data_path './data/Urban100' --dataset 'Urban100_15' --test_sigma 15 --S_t 1 --R_t 1
python main_for_gaussian.py --data_path './data/Urban100' --dataset 'Urban100_25' --test_sigma 25 --S_t 1 --R_t 1
python main_for_gaussian.py --data_path './data/Urban100' --dataset 'Urban100_50' --test_sigma 50 --S_t 1 --R_t 1




#TABLE 2, Robust Gaussian image denoising:
#DMID-d:
#ImageNet
python main_for_gaussian.py --data_path './data/ImageNet' --dataset 'ImageNet_50_d' --test_sigma 50 --S_t 2 --R_t 500 --mmse_average True
python main_for_gaussian.py --data_path './data/ImageNet' --dataset 'ImageNet_100_d' --test_sigma 100 --S_t 2 --R_t 500 --mmse_average True
python main_for_gaussian.py --data_path './data/ImageNet' --dataset 'ImageNet_150_d' --test_sigma 150 --S_t 2 --R_t 500 --mmse_average True
python main_for_gaussian.py --data_path './data/ImageNet' --dataset 'ImageNet_200_d' --test_sigma 200 --S_t 2 --R_t 500 --mmse_average True
python main_for_gaussian.py --data_path './data/ImageNet' --dataset 'ImageNet_250_d' --test_sigma 250 --S_t 2 --R_t 500 --mmse_average True
#CBSD68
python main_for_gaussian.py --data_path './data/CBSD68' --dataset 'CBSD68_50_d' --test_sigma 50 --S_t 2 --R_t 500 --mmse_average True
python main_for_gaussian.py --data_path './data/CBSD68' --dataset 'CBSD68_100_d' --test_sigma 100 --S_t 2 --R_t 500 --mmse_average True
python main_for_gaussian.py --data_path './data/CBSD68' --dataset 'CBSD68_150_d' --test_sigma 150 --S_t 2 --R_t 500 --mmse_average True
python main_for_gaussian.py --data_path './data/CBSD68' --dataset 'CBSD68_200_d' --test_sigma 200 --S_t 2 --R_t 500 --mmse_average True
python main_for_gaussian.py --data_path './data/CBSD68' --dataset 'CBSD68_250_d' --test_sigma 250 --S_t 2 --R_t 500 --mmse_average True
#Kodak24
python main_for_gaussian.py --data_path './data/Kodak24' --dataset 'Kodak24_50_d' --test_sigma 50 --S_t 2 --R_t 500 --mmse_average True
python main_for_gaussian.py --data_path './data/Kodak24' --dataset 'Kodak24_100_d' --test_sigma 100 --S_t 2 --R_t 500 --mmse_average True
python main_for_gaussian.py --data_path './data/Kodak24' --dataset 'Kodak24_150_d' --test_sigma 150 --S_t 2 --R_t 500 --mmse_average True
python main_for_gaussian.py --data_path './data/Kodak24' --dataset 'Kodak24_200_d' --test_sigma 200 --S_t 2 --R_t 500 --mmse_average True
python main_for_gaussian.py --data_path './data/Kodak24' --dataset 'Kodak24_250_d' --test_sigma 250 --S_t 2 --R_t 500 --mmse_average True
#McMaster
python main_for_gaussian.py --data_path './data/McMaster' --dataset 'McMaster_50_d' --test_sigma 50 --S_t 2 --R_t 500 --mmse_average True
python main_for_gaussian.py --data_path './data/McMaster' --dataset 'McMaster_100_d' --test_sigma 100 --S_t 2 --R_t 500 --mmse_average True
python main_for_gaussian.py --data_path './data/McMaster' --dataset 'McMaster_150_d' --test_sigma 150 --S_t 2 --R_t 500 --mmse_average True
python main_for_gaussian.py --data_path './data/McMaster' --dataset 'McMaster_200_d' --test_sigma 200 --S_t 2 --R_t 500 --mmse_average True
python main_for_gaussian.py --data_path './data/McMaster' --dataset 'McMaster_250_d' --test_sigma 250 --S_t 2 --R_t 500 --mmse_average True


#DMID-p: the sampling times S_t are searched over 6 choices: N, N//2, N//3, N//4, N//5. Under these choices, DMID-p performs well on LPIPS.
#ImageNet
python main_for_gaussian.py --data_path './data/ImageNet' --dataset 'ImageNe_50_p' --test_sigma 50 --S_t 57 --R_t 1
python main_for_gaussian.py --data_path './data/ImageNet' --dataset 'ImageNet_100_p' --test_sigma 100 --S_t 107 --R_t 1
python main_for_gaussian.py --data_path './data/ImageNet' --dataset 'ImageNet_150_p' --test_sigma 150 --S_t 145 --R_t 1
python main_for_gaussian.py --data_path './data/ImageNet' --dataset 'ImageNet_200_p' --test_sigma 200 --S_t 173 --R_t 1
python main_for_gaussian.py --data_path './data/ImageNet' --dataset 'ImageNet_250_p' --test_sigma 250 --S_t 196 --R_t 1
#CBSD68
python main_for_gaussian.py --data_path './data/CBSD68' --dataset 'CBSD68_50_p' --test_sigma 50 --S_t 28 --R_t 1
python main_for_gaussian.py --data_path './data/CBSD68' --dataset 'CBSD68_100_p' --test_sigma 100 --S_t 43 --R_t 1
python main_for_gaussian.py --data_path './data/CBSD68' --dataset 'CBSD68_150_p' --test_sigma 150 --S_t 72 --R_t 1
python main_for_gaussian.py --data_path './data/CBSD68' --dataset 'CBSD68_200_p' --test_sigma 200 --S_t 69 --R_t 1
python main_for_gaussian.py --data_path './data/CBSD68' --dataset 'CBSD68_250_p' --test_sigma 250 --S_t 98 --R_t 1
#Kodak24
python main_for_gaussian.py --data_path './data/Kodak24' --dataset 'Kodak24_50_p' --test_sigma 50 --S_t 57 --R_t 1
python main_for_gaussian.py --data_path './data/Kodak24' --dataset 'Kodak24_100_p' --test_sigma 100 --S_t 107 --R_t 1
python main_for_gaussian.py --data_path './data/Kodak24' --dataset 'Kodak24_150_p' --test_sigma 150 --S_t 97 --R_t 1
python main_for_gaussian.py --data_path './data/Kodak24' --dataset 'Kodak24_200_p' --test_sigma 200 --S_t 86 --R_t 1
python main_for_gaussian.py --data_path './data/Kodak24' --dataset 'Kodak24_250_p' --test_sigma 250 --S_t 98 --R_t 1
#McMaster
python main_for_gaussian.py --data_path './data/McMaster' --dataset 'McMaster_50_p' --test_sigma 50 --S_t 57 --R_t 1
python main_for_gaussian.py --data_path './data/McMaster' --dataset 'McMaster_100_p' --test_sigma 100 --S_t 107 --R_t 1
python main_for_gaussian.py --data_path './data/McMaster' --dataset 'McMaster_150_p' --test_sigma 150 --S_t 143 --R_t 1
python main_for_gaussian.py --data_path './data/McMaster' --dataset 'McMaster_200_p' --test_sigma 200 --S_t 173 --R_t 1
python main_for_gaussian.py --data_path './data/McMaster' --dataset 'McMaster_250_p' --test_sigma 250 --S_t 196 --R_t 1





#TABLE 3, real-world image denoising:

#cc-d
python main_for_real.py --clean_path './data/CC/GT' --noisy_path './pre-trained/CC.pt' --datatype 'CC_d' --S_t 2 --R_t 500
#cc-p:
python main_for_real.py --clean_path './data/CC/GT' --noisy_path './pre-trained/CC.pt' --datatype 'CC_p' --S_t 1 --R_t 1
#polyu-d
python main_for_real.py --clean_path './data/PolyU/GT' --noisy_path './pre-trained/PolyU.pt' --datatype 'PolyU_d' --S_t 3 --R_t 333
#polyu-p:
python main_for_real.py --clean_path './data/PolyU/GT' --noisy_path './pre-trained/PolyU.pt' --datatype 'PolyU_p' --S_t 1 --R_t 1
#fmdd-d
python main_for_real.py --clean_path './data/FMDD/GT' --noisy_path './pre-trained/FMDD.pt' --datatype 'FMDD_d' --S_t 3 --R_t 333
#fmdd-p:
python main_for_real.py --clean_path './data/FMDD/GT' --noisy_path './pre-trained/FMDD.pt' --datatype 'FMDD_p' --S_t 2 --R_t 1

