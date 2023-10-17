#If you want to reproduce quickly, you can reduce the repetitions times R_t. For example, setting R_ T=1 or R_ T=10, the approximate results can be obtained.

#TABLE 1, Classical Gaussian image denoising:

#CBSD68
python main_for_gaussian.py --data_path './data/CBSD68' --dataset 'CBSD68' --test_sigma 15 --S_t 2 --R_t 500
python main_for_gaussian.py --data_path './data/CBSD68' --dataset 'CBSD68' --test_sigma 25 --S_t 2 --R_t 500
python main_for_gaussian.py --data_path './data/CBSD68' --dataset 'CBSD68' --test_sigma 50 --S_t 2 --R_t 500
#Kodak24
python main_for_gaussian.py --data_path './data/Kodak24' --dataset 'Kodak24' --test_sigma 15 --S_t 2 --R_t 500
python main_for_gaussian.py --data_path './data/Kodak24' --dataset 'Kodak24' --test_sigma 25 --S_t 2 --R_t 500
python main_for_gaussian.py --data_path './data/Kodak24' --dataset 'Kodak24' --test_sigma 50 --S_t 2 --R_t 500
#McMaster
python main_for_gaussian.py --data_path './data/McMaster' --dataset 'McMaster' --test_sigma 15 --S_t 2 --R_t 500
python main_for_gaussian.py --data_path './data/McMaster' --dataset 'McMaster' --test_sigma 25 --S_t 2 --R_t 500
python main_for_gaussian.py --data_path './data/McMaster' --dataset 'McMaster' --test_sigma 50 --S_t 2 --R_t 500
#Urban100
python main_for_gaussian.py --data_path './data/Urban100' --dataset 'Urban100' --test_sigma 15 --S_t 1 --R_t 1
python main_for_gaussian.py --data_path './data/Urban100' --dataset 'Urban100' --test_sigma 25 --S_t 1 --R_t 1
python main_for_gaussian.py --data_path './data/Urban100' --dataset 'Urban100' --test_sigma 50 --S_t 1 --R_t 1

#TABLE 2, Robust Gaussian image denoising:
#DMID-d:
#ImageNet
python main_for_gaussian.py --data_path './data/ImageNet' --dataset 'ImageNet' --test_sigma 50 --S_t 1 --R_t 1
python main_for_gaussian.py --data_path './data/ImageNet' --dataset 'ImageNet' --test_sigma 100 --S_t 1 --R_t 1
python main_for_gaussian.py --data_path './data/ImageNet' --dataset 'ImageNet' --test_sigma 150 --S_t 1 --R_t 1
python main_for_gaussian.py --data_path './data/ImageNet' --dataset 'ImageNet' --test_sigma 200 --S_t 1 --R_t 1
python main_for_gaussian.py --data_path './data/ImageNet' --dataset 'ImageNet' --test_sigma 250 --S_t 1 --R_t 1
#CBSD68
python main_for_gaussian.py --data_path './data/CBSD68' --dataset 'CBSD68' --test_sigma 50 --S_t 2 --R_t 500
python main_for_gaussian.py --data_path './data/CBSD68' --dataset 'CBSD68' --test_sigma 100 --S_t 2 --R_t 500
python main_for_gaussian.py --data_path './data/CBSD68' --dataset 'CBSD68' --test_sigma 150 --S_t 2 --R_t 500
python main_for_gaussian.py --data_path './data/CBSD68' --dataset 'CBSD68' --test_sigma 200 --S_t 2 --R_t 500
python main_for_gaussian.py --data_path './data/CBSD68' --dataset 'CBSD68' --test_sigma 250 --S_t 2 --R_t 500
#Kodak24
python main_for_gaussian.py --data_path './data/Kodak24' --dataset 'Kodak24' --test_sigma 50 --S_t 2 --R_t 500
python main_for_gaussian.py --data_path './data/Kodak24' --dataset 'Kodak24' --test_sigma 100 --S_t 2 --R_t 500
python main_for_gaussian.py --data_path './data/Kodak24' --dataset 'Kodak24' --test_sigma 150 --S_t 2 --R_t 500
python main_for_gaussian.py --data_path './data/Kodak24' --dataset 'Kodak24' --test_sigma 200 --S_t 2 --R_t 500
python main_for_gaussian.py --data_path './data/Kodak24' --dataset 'Kodak24' --test_sigma 250 --S_t 2 --R_t 500
#McMaster
python main_for_gaussian.py --data_path './data/McMaster' --dataset 'McMaster' --test_sigma 50 --S_t 2 --R_t 500
python main_for_gaussian.py --data_path './data/McMaster' --dataset 'McMaster' --test_sigma 100 --S_t 2 --R_t 500
python main_for_gaussian.py --data_path './data/McMaster' --dataset 'McMaster' --test_sigma 150 --S_t 2 --R_t 500
python main_for_gaussian.py --data_path './data/McMaster' --dataset 'McMaster' --test_sigma 200 --S_t 2 --R_t 500
python main_for_gaussian.py --data_path './data/McMaster' --dataset 'McMaster' --test_sigma 250 --S_t 1 --R_t 1

#DMID-p: the sampling times S_t are searched over 6 choices: N, N//2, N//3, N//4, N//5. Under these choices, DMID-p performs well on LPIPS.
#ImageNet
python main_for_gaussian.py --data_path './data/ImageNet' --dataset 'ImageNet' --test_sigma 50 --S_t 57 --R_t 1
python main_for_gaussian.py --data_path './data/ImageNet' --dataset 'ImageNet' --test_sigma 100 --S_t 107 --R_t 1
python main_for_gaussian.py --data_path './data/ImageNet' --dataset 'ImageNet' --test_sigma 150 --S_t 145 --R_t 1
python main_for_gaussian.py --data_path './data/ImageNet' --dataset 'ImageNet' --test_sigma 200 --S_t 173 --R_t 1
python main_for_gaussian.py --data_path './data/ImageNet' --dataset 'ImageNet' --test_sigma 250 --S_t 196 --R_t 1
#CBSD68
python main_for_gaussian.py --data_path './data/CBSD68' --dataset 'CBSD68' --test_sigma 50 --S_t 28 --R_t 1
python main_for_gaussian.py --data_path './data/CBSD68' --dataset 'CBSD68' --test_sigma 100 --S_t 43 --R_t 1
python main_for_gaussian.py --data_path './data/CBSD68' --dataset 'CBSD68' --test_sigma 150 --S_t 72 --R_t 1
python main_for_gaussian.py --data_path './data/CBSD68' --dataset 'CBSD68' --test_sigma 200 --S_t 69 --R_t 1
python main_for_gaussian.py --data_path './data/CBSD68' --dataset 'CBSD68' --test_sigma 250 --S_t 98 --R_t 1
#Kodak24
python main_for_gaussian.py --data_path './data/Kodak24' --dataset 'Kodak24' --test_sigma 50 --S_t 57 --R_t 1
python main_for_gaussian.py --data_path './data/Kodak24' --dataset 'Kodak24' --test_sigma 100 --S_t 107 --R_t 1
python main_for_gaussian.py --data_path './data/Kodak24' --dataset 'Kodak24' --test_sigma 150 --S_t 97 --R_t 1
python main_for_gaussian.py --data_path './data/Kodak24' --dataset 'Kodak24' --test_sigma 200 --S_t 86 --R_t 1
python main_for_gaussian.py --data_path './data/Kodak24' --dataset 'Kodak24' --test_sigma 250 --S_t 98 --R_t 1
#McMaster
python main_for_gaussian.py --data_path './data/McMaster' --dataset 'McMaster' --test_sigma 50 --S_t 57 --R_t 1
python main_for_gaussian.py --data_path './data/McMaster' --dataset 'McMaster' --test_sigma 100 --S_t 107 --R_t 1
python main_for_gaussian.py --data_path './data/McMaster' --dataset 'McMaster' --test_sigma 150 --S_t 143 --R_t 1
python main_for_gaussian.py --data_path './data/McMaster' --dataset 'McMaster' --test_sigma 200 --S_t 173 --R_t 1
python main_for_gaussian.py --data_path './data/McMaster' --dataset 'McMaster' --test_sigma 250 --S_t 196 --R_t 1

#TABLE 3, real-world image denoising:

#cc-d
python main_for_real.py --clean_path '' --noisy_path './data/CC-full/Noisy/' --datatype 'CC' --pertrianed './pre-trained/CC.pt' --S_t 2 --R_t 500
#cc-p:
python main_for_real.py --clean_path '' --noisy_path './data/CC-full/Noisy/' --datatype 'CC' --pertrianed './pre-trained/CC.pt' --S_t 1 --R_t 1
#polyu-d
python main_for_real.py --clean_path '' --noisy_path '' --datatype 'PolyU' --pertrianed './pre-trained/PolyU.pt' --S_t 3 --R_t 333
#polyu-p:
python main_for_real.py --clean_path '' --noisy_path '' --datatype 'PolyU' --pertrianed './pre-trained/PolyU.pt' --S_t 1 --R_t 1
#fmdd-d
python main_for_real.py --clean_path '' --noisy_path '' --datatype 'FMDD' --pertrianed './pre-trained/FMDD.pt' --S_t 3 --R_t 333
#fmdd-p:
python main_for_real.py --clean_path '' --noisy_path '' --datatype 'FMDD' --pertrianed './pre-trained/FMDD.pt' --S_t 2 --R_t 1

