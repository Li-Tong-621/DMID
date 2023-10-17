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

