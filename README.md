# Stimulating the Diffusion Model for Image Denoising via Adaptive Embedding and Ensembling

<hr />

>**Abstract:** *Image denoising is a fundamental problem in computational photography, where achieving high perception with low distortion is highly demanding. Current methods either struggle with perceptual quality or suffer from significant distortion. Recently, the emerging diffusion model has achieved state-of-the-art performance in various tasks and demonstrates great potential for image denoising. However, stimulating diffusion models for image denoising is not straightforward and requires solving several critical problems. For one thing, the input inconsistency hinders the connection between diffusion models and image denoising. For another, the content inconsistency between the generated image and the desired denoised image introduces distortion. To tackle these problems, we present a novel strategy called the Diffusion Model for Image Denoising (DMID) by understanding and rethinking the diffusion model from a denoising perspective. Our DMID strategy includes an adaptive embedding method that embeds the noisy image into a pre-trained diffusion model and an adaptive ensembling method that reduces distortion in the denoised image. Our DMID strategy achieves state-of-the-art performance on both distortion-based and perception-based metrics, for both Gaussian and real-world image denoising.*
<hr />


## Quick Start
```
python main_for_gaussian.py
```
```
python main_for_real.py
```

## Evaluation
- Download the pre-trained Diffusion [model](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt)(from [guided-diffusion](https://github.com/openai/guided-diffusion)) and place it in `./pre-trained/`.
- Download testsets (CBSD68, Kodak24, McMaster, Urban100, ImageNet, CC, PolyU, FMDD).

#### Gaussian image denoising testing
- To obtain denoised images, run
```
python main_for_gaussian.py --data_path your_data_path --dataset test_dataset_name --test_sigma test_noise_level --S_t Sampling_times --R_t Repetition_times
```


#### Real-world image denoising testing
- To obtain denoised images, run
```
python main_for_real.py --clean_path clean_data_path --noisy_path noisy_data_path --datatype test_dataset_name --pertrianed latent_images_path --objective objective
```

- To quickly reproduce the reported results of CC, run
```
python main_for_real.py --clean_path './data/CC-full/GT/' --noisy_path './data/cc-full/Noisy/' --datatype 'cc' --pertrianed './pre-trained/cc-latent.pt' --objective 'DMID-p'
```
```
python main_for_real.py --clean_path './data/CC-full/GT/' --noisy_path './data/cc-full/Noisy/' --datatype 'cc' --pertrianed './pre-trained/cc-latent.pt' --objective 'DMID-d'
```

## Results

result tables can be found in `./images/`.
