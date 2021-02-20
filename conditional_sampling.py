'''
sampling.py: 
Converts low resolution face images to high resolution face images through conditional sampling of DDPM.
You don't need to modify except for the main function.

Please modify the things below.
1. gpu type
2. what kinds of degradation type you will use which is located in degrade.py
'''
import os
import cv2
import glob
import json
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from lib.model import UNet
from lib.diffusion import GaussianDiffusion, make_beta_schedule

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torchvision.transforms as T
import torchvision.datasets
import numpy as np
from torch.utils.data import Subset

from degrade import *


'''
You should specify the gpu.
'''
torch.cuda.set_device(1)


'''
class for DDPM models
'''
class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, obj(b) if isinstance(b, dict) else b)

class DDP(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()

        self.conf  = conf
        self.save_hyperparameters()

        self.model = UNet(self.conf.model.in_channel,
                          self.conf.model.channel,
                          channel_multiplier=self.conf.model.channel_multiplier,
                          n_res_blocks=self.conf.model.n_res_blocks,
                          attn_strides=self.conf.model.attn_strides,
                          dropout=self.conf.model.dropout,
                          fold=self.conf.model.fold,
                          )

        self.ema   = UNet(self.conf.model.in_channel,
                          self.conf.model.channel,
                          channel_multiplier=self.conf.model.channel_multiplier,
                          n_res_blocks=self.conf.model.n_res_blocks,
                          attn_strides=self.conf.model.attn_strides,
                          dropout=self.conf.model.dropout,
                          fold=self.conf.model.fold,
                          )

        self.betas = make_beta_schedule(schedule=self.conf.model.schedule.type,
                                        start=self.conf.model.schedule.beta_start,
                                        end=self.conf.model.schedule.beta_end,
                                        n_timestep=self.conf.model.schedule.n_timestep)

        self.diffusion = GaussianDiffusion(betas=self.betas,
                                           model_mean_type=self.conf.model.mean_type,
                                           model_var_type=self.conf.model.var_type,
                                           loss_type=self.conf.model.loss_type)



    def setup(self, stage):

        self.train_set, self.valid_set = dataset.get_train_data(self.conf)

    def forward(self, x):

        return self.diffusion.p_sample_loop(self.model, x.shape)

    def configure_optimizers(self):

        if self.conf.training.optimizer.type == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.conf.training.optimizer.lr)
        else:
            raise NotImplementedError

        return optimizer

    def training_step(self, batch, batch_nb):

        img, _ = batch
        time   = (torch.rand(img.shape[0]) * 1000).type(torch.int64).to(img.device)
        loss   = self.diffusion.training_losses(self.model, img, time).mean()

        accumulate(self.ema, self.model.module if isinstance(self.model, nn.DataParallel) else self.model, 0.9999)

        tensorboard_logs = {'train_loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}

    def train_dataloader(self):

        train_loader = DataLoader(self.train_set,
                                  batch_size=self.conf.training.dataloader.batch_size,
                                  shuffle=True,
                                  num_workers=self.conf.training.dataloader.num_workers,
                                  pin_memory=True,
                                  drop_last=self.conf.training.dataloader.drop_last)

        return train_loader

    def validation_step(self, batch, batch_nb):

        img, _ = batch
        time   = (torch.rand(img.shape[0]) * 1000).type(torch.int64).to(img.device)
        loss   = self.diffusion.training_losses(self.ema, img, time).mean()

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):

        avg_loss         = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}

        shape  = (16, 3, self.conf.dataset.resolution, self.conf.dataset.resolution)
        sample = progressive_samples_fn(self.ema, self.diffusion, shape, device='cuda' if self.on_gpu else 'cpu')

        grid = make_grid(sample['samples'], nrow=4)
        self.logger.experiment.add_image(f'generated_images', grid, self.current_epoch)

        grid = make_grid(sample['progressive_samples'].reshape(-1, 3, self.conf.dataset.resolution, self.conf.dataset.resolution), nrow=20)
        self.logger.experiment.add_image(f'progressive_generated_images', grid, self.current_epoch)
        
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def val_dataloader(self):
        valid_loader = DataLoader(self.valid_set,
                                  batch_size=self.conf.validation.dataloader.batch_size,
                                  shuffle=False,
                                  num_workers=self.conf.validation.dataloader.num_workers,
                                  pin_memory=True,
                                  drop_last=self.conf.validation.dataloader.drop_last)

        return valid_loader

'''
sampling function:
    1. condtional sampling
    2. interpolate sampling
'''
def progressive_conditional_samples_fn(model, diffusion, shape, device, img_start, t_start, include_x0_pred_freq=50):
    samples, progressive_samples = diffusion.p_sample_loop_progressive_conditional(
        model=model,
        shape=shape,
        noise_fn=torch.randn,
        img_start = img_start,
        device=device,
        t_start = t_start,
        include_x0_pred_freq=include_x0_pred_freq
    )
    return {'samples': (samples + 1)/2, 'progressive_samples': (progressive_samples + 1)/2}

def interpolate_samples_fn(model, diffusion, image_1, image_2, t, lam = 0.5):
    interpolate_img = diffusion.interpolate(
        model=model,
        image_1 = image_1,
        image_2 = image_2,
        t = t,
        lam = lam
    )
    return (interpolate_img + 1)/2

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.")

    # Eval specific args
    parser.add_argument("--model_dir", type=str, default='./celeba.ckpt', help="Path to model for loading.")
    parser.add_argument("--sample_dir", type=str, default='samples', help="Path to save generated samples.")
    parser.add_argument("--prog_sample_freq", type=int, default=200, help="Progressive sample frequency.")

    # Dataset args
    parser.add_argument("--original_dir", type=str, required=True, default='data128x128', help="Path to celebA 128 x 128 cropped images.")
    parser.add_argument("--degraded_dir", type=str, required=True, default='data128x128_bicubic_2', help="Path to downgraded celebA 128 x 128 cropped images.")

    args = parser.parse_args()

    path_to_config = args.config
    with open(path_to_config, 'r') as f:
        conf = json.load(f)

    conf = obj(conf)

    
    denoising_diffusion_model = DDP(conf)

    denoising_diffusion_model.cuda()
    state_dict = torch.load(args.model_dir)
    denoising_diffusion_model.load_state_dict(state_dict['state_dict'])
    denoising_diffusion_model.eval()

    device = 'cuda'
    if not os.path.exists(args.sample_dir):
        os.mkdir(args.sample_dir)

    degraded_imgs = sorted(glob.glob(args.degraded_dir+"/*"))
    original_imgs = sorted(glob.glob(args.original_dir+"/*"))
    assert len(original_imgs) == len(degraded_imgs)

    if len(degraded_imgs) > 25:
        repeat = len(degraded_imgs) // 25 +1
        last_batch = len(degraded_imgs) - (repeat-1)*25 
    else:
        repeat = 1
        last_batch = len(degraded_imgs)

    # progressive sampling
    # for bicubic
    starts = [300, 300, 250]

    # for bicubic 3
    #starts = [400, 400, 330]

    # for gaussian 1
    #starts = [330]

    # for gaussian 2
    #starts = [400]

    # for gaussian 3
    #starts = [400, 400, 330]

    # for motion
    #starts = [300, 400]

    batch = 25
    for count in range(repeat):

        batch_saved = batch

        if count == repeat-1:
            batch = last_batch

        img = torch.zeros((batch, 3, conf.dataset.resolution, conf.dataset.resolution), dtype=torch.float32).to(device)
        originals = []
        blurs = []

        # progressive sampling
        samples = []

        for i in range(batch):

            original = plt.imread(original_imgs[i+batch*count])[:, :, :3]
            blur_img = plt.imread(degraded_imgs[i+batch*count])[:, :, :3]
            blur_img = cv2.resize(blur_img, dsize=(128, 128))

            blurs.append(blur_img)
            originals.append(original)
                
            # to cuda tensor
                
            blur_img_c = blur_img*2-1
            img[i] = torch.Tensor(blur_img_c.transpose(2, 0, 1))


        # model mean type: eps

        # first sampling
        sample = progressive_conditional_samples_fn(denoising_diffusion_model.ema,
                                            denoising_diffusion_model.diffusion,
                                            (batch, 3, conf.dataset.resolution, conf.dataset.resolution),
                                            device='cuda',
                                            img_start = img,
                                            t_start = 1000-starts[0], 
                                            include_x0_pred_freq=args.prog_sample_freq)
        samples.append(sample)

        for i in range(batch):

            img = samples[0]['samples'][i]
            #plt.imsave(os.path.join(args.sample_dir, f'sample_1_{i}.png'), img.cpu().numpy().transpose(1, 2, 0))

            img = samples[0]['progressive_samples'][i]
            img = make_grid(img, nrow=args.prog_sample_freq)
            #plt.imsave(os.path.join(args.sample_dir, f'prog_sample_0_{i}.png'), img.clamp(min=0, max=1).cpu().numpy().transpose(1, 2, 0))

        #print("[i]:", str(1), "th sampling finished")

        
        # repeated sampling
        for j in range(len(starts)-1):
            img = torch.zeros((batch, 3, conf.dataset.resolution, conf.dataset.resolution), dtype=torch.float32).to(device)
            for i in range(batch):
                tmp_img = (samples[j]['samples'][i].cpu().numpy())*2-1

                # change the following script (interpolate or not) to fit your needs    
                if False:
                    #print("interpolate")
                    lam = 0.2
                    interpolate = blurs[i].transpose(2,0,1)*2-1
                    img[i] = torch.Tensor( (1-lam)*tmp_img + lam*interpolate )
                else:
                    img[i] = torch.Tensor(tmp_img)    
                
            sample = progressive_conditional_samples_fn(denoising_diffusion_model.ema,
                                                denoising_diffusion_model.diffusion,
                                                (batch, 3, conf.dataset.resolution, conf.dataset.resolution),
                                                device='cuda',
                                                img_start = img,
                                                t_start = 1000-starts[j+1], 
                                                include_x0_pred_freq=args.prog_sample_freq)
            samples.append(sample)

            for i in range(batch):

                img = samples[j+1]['samples'][i]
                #plt.imsave(os.path.join(args.sample_dir, f'sample_{j+1}_{i}.png'), img.cpu().numpy().transpose(1, 2, 0))

                img = samples[j+1]['progressive_samples'][i]
                img = make_grid(img, nrow=args.prog_sample_freq)
                #plt.imsave(os.path.join(args.sample_dir, f'prog_sample_{j+1}_{i}.png'), img.clamp(min=0, max=1).cpu().numpy().transpose(1, 2, 0))

            #print("[i]: ", j+2, "th sampling finished")

        file_name_concat = str(args.degraded_dir)+ "_rep_" + str(len(starts))
        for item in starts:
            file_name_concat += "_"+str(item)
            
        if not os.path.exists(os.path.join(args.sample_dir, file_name_concat)):
            os.mkdir(os.path.join(args.sample_dir, file_name_concat))

       # print all progressive sampling results
        all = []
        all_without_original = []
        prog_samples_all = torch.zeros((batch, len(starts)+2, 3, conf.dataset.resolution, conf.dataset.resolution), dtype=torch.float32).to(device)
        prog_samples_all_without_original = torch.zeros((batch, len(starts)+1, 3, conf.dataset.resolution, conf.dataset.resolution), dtype=torch.float32).to(device)
        for i in range(batch):
            for j in range(len(starts)):
                prog_samples_all[i, j, :, :, :] = samples[j]['samples'][i]
                prog_samples_all_without_original[i, j, :, :, :] = samples[j]['samples'][i]
            plt.imsave(os.path.join(args.sample_dir, file_name_concat, f"{batch_saved*count+i}.png"), prog_samples_all[i, len(starts)-1, :, :, :].clamp(min=0, max=1).cpu().numpy().transpose(1, 2, 0)) 
            prog_samples_all[i, len(starts), :, :, :] = torch.Tensor(blurs[i].transpose(2, 0, 1))
            prog_samples_all_without_original[i, len(starts), :, :, :] = torch.Tensor(blurs[i].transpose(2, 0, 1))
            prog_samples_all[i, len(starts)+1, :, :, :] = torch.Tensor(originals[i].transpose(2, 0, 1))
                
            img = make_grid(prog_samples_all[i], nrow=len(starts)+2)
            all.append(img)
            img = make_grid(prog_samples_all_without_original[i], nrow=len(starts)+1)
            all_without_original.append(img)
            
        all = torch.cat(all, dim = 1)
        all_without_original = torch.cat(all_without_original, dim = 1)
            
        plt.imsave(os.path.join(args.sample_dir, f'{file_name_concat}_({count})_all.png'), all.clamp(min=0, max=1).cpu().numpy().transpose(1, 2, 0))
        plt.imsave(os.path.join(args.sample_dir, f'{file_name_concat}_({count})_wo_original.png'), all_without_original.clamp(min=0, max=1).cpu().numpy().transpose(1, 2, 0))
        
        print(f"######## {count+1}th batch sampling finished ########\n")

    print(f'######## All sampling finished! ########\n')