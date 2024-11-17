import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
from torchinfo import summary
from torchvision import models
import torchvision.utils as vutils
import torchvision.transforms.functional as F

from preprocess.ddim_preprocessor import DDIMDataPreprocessor
import metrics

from logger.main_logger import MainLogger
from logger.utils import time_to_str

from datetime import datetime
import os
from tqdm import tqdm
import time
import pickle

import numpy as np

import argparse
import matplotlib.pyplot as plt
from train.trainer import Trainer

from utils import get_default_device
from model.DCGAN import weights_init
from metrics import Metrics

from model.DDIM import Model, EMAHelper

import math


class DDIMTrainer(Trainer):
    def __init__(self, 
                 args: argparse.Namespace,
                 data_pre: DDIMDataPreprocessor):
        self.logger = MainLogger(args)
        self.device = get_default_device()

        self.data_pre = data_pre
        self.train_loader, metric_loader = self.data_pre.get_data_loader()
        self.metric = Metrics(metric_loader)
        
        betas = np.linspace(
            0.0001, 0.02, 1000, dtype=np.float64
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]
        self.logvar = betas.log()
        
        if args.model_path != '':
            datetime_now = args.model_path
        else:
            datetime_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_save_path = os.path.join('.', 'save', 'dcgan', datetime_now)
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        self.logger.debug(f'save path: {self.model_save_path}')

    
    def save_model(self, typ, states, iters, value, images):
        save_path = os.path.join(self.model_save_path, typ)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        for filename in os.listdir(save_path):
            file_path = os.path.join(save_path, filename)
            if os.path.isfile(file_path) and filename.endswith('.pt'):
                os.remove(file_path)
        
        torch.save(states, os.path.join(save_path, f'{iters}_{value:.04f}.pt'))
        
        plt.clf()
        plt.axis("off")
        plt.title("fake images")
        plt.imshow(np.transpose(vutils.make_grid(images, padding=2, normalize=True),(1,2,0)))
        plt.savefig(os.path.join(save_path, f'{iters}_fake_image.png'))
        
        self.logger.debug(f'{iters} model save')
        
    
    # def load_model(self, typ):
    #     saved_state = torch.load(os.path.join(self.model_save_path, typ, f'{iters}.pt'))
    #     self.model_g.load_state_dict(saved_state['model_g'])
    #     self.optimizer_g.load_state_dict(saved_state['optimizer_g'])    
    #     self.model_d.load_state_dict(saved_state['model_d'])
    #     self.optimizer_d.load_state_dict(saved_state['optimizer_d'])   
    
    
    def noise_estimation_loss(self, model, x0, t,  e, b, keepdim=False):
        a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
        output = model(x, t.float())
        if keepdim:
            return (e - output).square().sum(dim=(1, 2, 3))
        else:
            return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)
        
        
    def compute_alpha(self, beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a
    
    
    def train(self):
        train_loader = self.train_loader
        model = Model().to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=0.0002)

        ema_helper = EMAHelper(mu=0.9999)
        ema_helper.register(model)

        start_epoch, step = 0, 0
        
        fixed_noise = torch.randn(50000, 3, 32, 32, device=self.device)

        iters = 0
        for epoch in range(start_epoch, 10000):
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)
                x = 2 * x - 1.0
                e = torch.randn_like(x)
                b = self.betas

                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = self.noise_estimation_loss(model, x, t, e, b)

                if iters % 100 == 0:
                    self.logger.debug(f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}")

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                except Exception:
                    pass
                optimizer.step()

                ema_helper.update(model)

                if (iters % 500 == 0) or ((epoch == 10000 - 1) and (i == len(train_loader) - 1)):
                    model.eval()
                    total_n_samples = 64
                    smaples_batch = 256
                    test_iter = int(total_n_samples / smaples_batch) + (1 if total_n_samples % smaples_batch > 0 else 0)
                    
                    generated = []
                    
                    for start in tqdm(range(test_iter), desc="test"):
                        end = min((start + 1) + smaples_batch, total_n_samples)
                        x = fixed_noise[start * smaples_batch:end]
                        skip = self.num_timesteps // 1000
                        seq = range(0, self.num_timesteps, skip)
                        
                        with torch.no_grad():
                            n = x.size(0)
                            seq_next = [-1] + list(seq[:-1])
                            x0_preds = []
                            xs = [x]
                            for i, j in zip(reversed(seq), reversed(seq_next)):
                                t = (torch.ones(n) * i).to(x.device)
                                next_t = (torch.ones(n) * j).to(x.device)
                                at = self.compute_alpha(self.betas, t.long())
                                at_next = self.compute_alpha(self.betas, next_t.long())
                                xt = xs[-1].to('cuda')
                                et = model(xt, t)
                                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                                x0_preds.append(x0_t.to('cpu'))
                                c1 = 0
                                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                                xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
                                xs.append(xt_next.to('cpu'))
                        self.logger.debug(len(xs))
                        
                        # x = self.sample_image(x, model)
                        x = (x + 1.0) / 2.0
                        x = torch.clamp(x, 0.0, 1.0)
                        generated.extend(x)

                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        ema_helper.state_dict(),
                        epoch,
                        step,
                    ]
                    
                    self.save_model('test', states, iters, 0, torch.FloatTensor(generated))
                    model.train()
                    
                iters += 1
                #     states = [
                #         model.state_dict(),
                #         optimizer.state_dict(),
                #         epoch,
                #         step,
                #     ]
                #     states.append(ema_helper.state_dict())

                #     torch.save(
                #         states,
                #         os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                #     )
                #     torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()
    
        
    # def train(self):
    #     real_images_loader = self.train_loader
        
    #     losses_g = []
    #     losses_d = []
        
    #     label_real = 0.9
    #     label_fake = 0.1
    #     iters = 0
    #     fixed_noise = torch.randn(64, 100, 1, 1, device=self.device)
        
    #     low_fid = 1e10
    #     high_is = 0
        
    #     real_batch = next(iter(real_images_loader))
    #     plt.axis("off")
    #     plt.title("real images")
    #     plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(self.device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
    #     plt.savefig(os.path.join(self.model_save_path, 'real_image.png'))
        
    #     start = time.time()

    #     self.logger.debug("train start")
    #     for epoch in range(self.epoch):
    #         for i, data in enumerate(real_images_loader):
    #             self.model_d.zero_grad()
                
    #             real_data = data[0].to(self.device)
    #             b_size = real_data.size(0)
    #             label = torch.full((b_size,), label_real, dtype=torch.float32, device=self.device)
    #             real_data = 0.9 * real_data + 0.1 * torch.randn((real_data.size()), device=self.device)
                
    #             output = self.model_d(real_data).view(-1)
    #             real_error_d = self.criterion(output, label)
    #             real_error_d.backward()
    #             x_d = output.mean().item()


    #             noise = torch.randn(b_size, 100, 1, 1, device=self.device)
    #             fake = self.model_g(noise)
    #             label.fill_(label_fake)
    #             fake = 0.9 * fake + 0.1 * torch.randn((fake.size()), device=self.device)
                
    #             output = self.model_d(fake.detach()).view(-1)
    #             fake_error_d = self.criterion(output, label)
    #             fake_error_d.backward()
    #             z1_gd = output.mean().item()

    #             gradient_penalty = self.compute_gradient_penalty(real_data, fake)
    #             error_d = real_error_d + fake_error_d + self.lambda_gp * gradient_penalty
    #             self.optimizer_d.step()

    #             self.model_g.zero_grad()
    #             label.fill_(label_real)
                
    #             output = self.model_d(fake).view(-1)
    #             error_g = self.criterion(output, label)
    #             error_g.backward()
    #             z2_gd = output.mean().item()
    #             self.optimizer_g.step()

    #             if i % 100 == 0:
    #                 self.logger.debug(f'[{epoch}/{self.epoch}][{i}/{len(real_images_loader)}]\tloss_d: {error_d:.4f}\tloss_g: {error_g:.4f}'
    #                                   + f'\tD(x): {x_d:.4f}\tD(G(z)): {z1_gd:.4f} / {z2_gd:.4f}')

    #             losses_g.append(error_g.item())
    #             losses_d.append(error_d.item())

    #             if (iters % 500 == 0) or ((epoch == self.epoch - 1) and (i == len(real_images_loader) - 1)):
    #                 with torch.no_grad():
    #                     fake = self.model_g(fixed_noise).detach().cpu()
                    
    #                 fake = 0.5 * fake + 0.5
    #                 fake = F.resize(fake, [299, 299])
    #                 inception_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    #                 inception_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    #                 fake = (fake - inception_mean) / inception_std
    #                 # fake = torch.utils.data.TensorDataset(fake)
                    
    #                 inception_score = self.metric.inception_score(torch.utils.data.DataLoader(fake, batch_size=64))
    #                 fid = self.metric.fid(torch.utils.data.DataLoader(fake, batch_size=64))
                    
    #                 self.logger.debug(f'inception score: {inception_score}\tfid: {fid}')
                    
    #                 if low_fid > fid:
    #                     low_fid = fid
    #                     self.logger.debug(f"{iters} lowest fid")
    #                     self.save_model('fid', iters, low_fid, fake)
    #                 if high_is < inception_score:
    #                     high_is = inception_score
    #                     self.logger.debug(f"{iters} highest is")
    #                     self.save_model('is', iters, high_is, fake)
                    
    #             iters += 1

    #     end = time.time()
    #     self.logger.debug(f'train finish\ttiem: {time_to_str(end - start)}')
        
    #     plt.clf()
    #     epoch_x = range(1, len(losses_g) + 1)
        
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(epoch_x, losses_d, label='Discriminator Loss')
    #     plt.plot(epoch_x, losses_g, label='Generator Loss')

    #     plt.title('Discriminator and Generator Loss')
    #     plt.xlabel('Iterations')
    #     plt.ylabel('Loss')
    #     plt.legend()
    #     plt.savefig(os.path.join(self.model_save_path, f'loss.png'))
