# Denosing Diffusion Generative Models

This is an unofficial implementation of ["Denoising Diffusion Probabilistic Models"](https://arxiv.org/abs/2006.11239) in [PyTorch(Lightning)](https://github.com/PyTorchLightning/pytorch-lightning). For a brief introduction to diffusion models, see [blog post](https://hmdolatabadi.github.io/posts/2020/09/ddp/).

<p align="center">
  <img width="522" height="132" src="/misc/DDP.gif">
</p>
<p align="center">
  <img width="680" height="342" src="/misc/cifar10.png">
</p>

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
conda env create --file DDPM.yaml
```

## Pre-trained Models and Datasets for CelebA 128x128

Pre-trained diffusion models on CelebA and Datasets for CelebA 128x128 (img_align_celeba.zip) can be found [here](https://drive.google.com/drive/folders/1LziVrfaoFZV6aUa7X9S768S_NS4SM-a9?usp=sharing).


## Training Diffusion Models

To train a diffusion model, first specify model architecture and hyperparameters in `config.json`. Once specified, run this command:

```train
python diffusion_lightning.py --train --config config.json --ckpt_dir PATH_TO_CHECKPOINTS --ckpt_freq CHECKPOINT_FREQ --n_gpu NUM_AVAIL_GPUS
```

## Sample Generation

To generate samples from a trained diffusion model specified by `config.json`, run this command:

```eval
python diffusion_lightning.py --config config.json --model_dir MODEL_DIRECTORY --sample_dir PATH_TO_SAVE_SAMPLES --n_samples NUM_SAMPLES
```

## Conditional Sample Generation (CelebA 128x128)

Before run below commands, 
1) Two files (celeba.ckpt, img_align_celeba) should be downloaded from the above google Drive.
2) Unzip img_align_celeba.zip file.
3) Please modify the below three scripts for your needs.

```make original celebA 128 x 128 test datasets 
python celebA.py
```

```make degraded celebA 128 x 128 test datasets 
python degrade.py
```

To generate samples from a trained diffusion model specified by `config.json`, run this command:
for example) python conditional_sampling.py --config config/diffusion_celeba.json --model_dir ./celeba.ckpt --sample_dir tmp --original_dir data128x128 --degraded_dir data128x128_motion_1

```eval
python conditional_sampling.py --config config.json --model_dir MODEL_DIRECTORY --sample_dir PATH_TO_SAVE_SAMPLES --original_dir PATH_TO_ORIGINAL_IMGS --degraded_dir PATH_TO_DEGRADED_IMGS
```

## CIFAR-10 FID Score

In the paper, the authors perform model selection using the FID score. Here, however, the model is only trained until 1000000 iterations and no model selection is performed due to limited computational resources. This way, we got an FID score of 5.1037.

## Acknowledgement

This repository is built upon the [official repository of diffusion models in TensorFlow](https://github.com/hojonathanho/diffusion) as well as parts of [this unofficial PyTorch implementation](https://github.com/rosinality/denoising-diffusion-pytorch) and [this unofficial PyTorch Lightening implementation](https://github.com/hmdolatabadi/denoising_diffusion).
