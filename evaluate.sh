#If you want to reproduce quickly, you can reduce the repetitions times R_t. For example, setting R_ T=1 or R_ T=10, the approximate results can be obtained.
#TABLE 1, Classical Gaussian image denoising:





#TABLE 3, real-world image denoising:

##cc-d
python main_for_real.py --clean_path './data/CC-full/GT/' --noisy_path './data/CC-full/Noisy/' --datatype 'CC' --pertrianed './pre-trained/CC.pt' --S_t 2 --R_t 500
#cc-p:
python main_for_real.py --clean_path './data/CC-full/GT/' --noisy_path './data/CC-full/Noisy/' --datatype 'CC' --pertrianed './pre-trained/CC.pt' --S_t 1 --R_t 1

##polyu-d
python main_for_real.py --clean_path '' --noisy_path '' --datatype 'PolyU' --pertrianed './pre-trained/PolyU.pt' --S_t 3 --R_t 333
#polyu-p:
python main_for_real.py --clean_path '' --noisy_path '' --datatype 'PolyU' --pertrianed './pre-trained/PolyU.pt' --S_t 1 --R_t 1

##fmdd-d
python main_for_real.py --clean_path '' --noisy_path '' --datatype 'FMDD' --pertrianed './pre-trained/FMDD.pt' --S_t 3 --R_t 333
#fmdd-p:
python main_for_real.py --clean_path '' --noisy_path '' --datatype 'FMDD' --pertrianed './pre-trained/FMDD.pt' --S_t 2 --R_t 1

