import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
from torchinfo import summary
from torchvision import models
import torchvision.utils as vutils
import torchvision.transforms.functional as F

from preprocess.dcgan_data_preprocessor import DCGANDataPreprocessor
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


class DCGANTrainer(Trainer):
    def __init__(self, 
                 args: argparse.Namespace,
                 model_g: nn.Module,
                 model_d: nn.Module,
                 data_pre: DCGANDataPreprocessor):
        self.logger = MainLogger(args)
        self.device = get_default_device()
        
        self.epoch = args.epoch
        self.max_lr = args.max_learning_rate
        
        self.model_g = model_g.to(self.device)
        self.model_d = model_d.to(self.device)
        
        self.lambda_gp = 10.0
        
        with torch.no_grad():
            self.logger.debug(f"Generator:\n{summary(self.model_g, input_size=(1, 100, 1, 1))}")
            self.logger.debug(f"Discriminator: {summary(self.model_d, input_size=(1, 3, 64, 64))}")
        self.model_g.apply(weights_init)
        self.model_d.apply(weights_init)

        self.data_pre = data_pre
        self.train_loader, metric_loader = self.data_pre.get_data_loader()
        self.metric = Metrics(metric_loader)
        
        self.optimizer_g = optim.Adam(self.model_g.parameters(), lr=self.max_lr, betas=[0.5, 0.999])
        self.optimizer_d = optim.Adam(self.model_d.parameters(), lr=self.max_lr, betas=[0.5, 0.999])
        
        self.criterion = nn.BCELoss()
        
        if args.model_path != '':
            datetime_now = args.model_path
        else:
            datetime_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_save_path = os.path.join('.', 'save', 'dcgan', datetime_now)
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        self.logger.debug(f'save path: {self.model_save_path}')
        
    
    def save_model(self, typ, iters, value, images):
        save_path = os.path.join(self.model_save_path, typ)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        for filename in os.listdir(save_path):
            file_path = os.path.join(save_path, filename)
            if os.path.isfile(file_path) and filename.endswith('.pt'):
                os.remove(file_path)
        
        torch.save({
            'model_g': self.model_g.state_dict(),
            'model_d': self.model_d.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict()
        }, os.path.join(save_path, f'{iters}_{value:.04f}.pt'))
        
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
    
    
    def compute_gradient_penalty(self, real_data, fake_data):
        alpha = torch.rand(real_data.size(0), 1, 1, 1, device=self.device)
        interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).requires_grad_(True)
    
        d_interpolates = self.model_d(interpolates)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates, device=self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty  
    
        
    def train(self):
        real_images_loader = self.train_loader
        
        losses_g = []
        losses_d = []
        
        label_real = 0.9
        label_fake = 0.1
        iters = 0
        fixed_noise = torch.randn(64, 100, 1, 1, device=self.device)
        
        low_fid = 1e10
        high_is = 0
        
        real_batch = next(iter(real_images_loader))
        plt.axis("off")
        plt.title("real images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(self.device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
        plt.savefig(os.path.join(self.model_save_path, 'real_image.png'))
        
        start = time.time()

        self.logger.debug("train start")
        for epoch in range(self.epoch):
            for i, data in enumerate(real_images_loader):
                self.model_d.zero_grad()
                
                real_data = data[0].to(self.device)
                b_size = real_data.size(0)
                label = torch.full((b_size,), label_real, dtype=torch.float32, device=self.device)
                real_data = 0.9 * real_data + 0.1 * torch.randn((real_data.size()), device=self.device)
                
                output = self.model_d(real_data).view(-1)
                real_error_d = self.criterion(output, label)
                real_error_d.backward()
                x_d = output.mean().item()


                noise = torch.randn(b_size, 100, 1, 1, device=self.device)
                fake = self.model_g(noise)
                label.fill_(label_fake)
                fake = 0.9 * fake + 0.1 * torch.randn((fake.size()), device=self.device)
                
                output = self.model_d(fake.detach()).view(-1)
                fake_error_d = self.criterion(output, label)
                fake_error_d.backward()
                z1_gd = output.mean().item()

                gradient_penalty = self.compute_gradient_penalty(real_data, fake)
                error_d = real_error_d + fake_error_d + self.lambda_gp * gradient_penalty
                self.optimizer_d.step()

                self.model_g.zero_grad()
                label.fill_(label_real)
                
                output = self.model_d(fake).view(-1)
                error_g = self.criterion(output, label)
                error_g.backward()
                z2_gd = output.mean().item()
                self.optimizer_g.step()

                if i % 100 == 0:
                    self.logger.debug(f'[{epoch}/{self.epoch}][{i}/{len(real_images_loader)}]\tloss_d: {error_d:.4f}\tloss_g: {error_g:.4f}'
                                      + f'\tD(x): {x_d:.4f}\tD(G(z)): {z1_gd:.4f} / {z2_gd:.4f}')

                losses_g.append(error_g.item())
                losses_d.append(error_d.item())

                if (iters % 500 == 0) or ((epoch == self.epoch - 1) and (i == len(real_images_loader) - 1)):
                    with torch.no_grad():
                        fake = self.model_g(fixed_noise).detach().cpu()
                    
                    fake = 0.5 * fake + 0.5
                    fake = F.resize(fake, [299, 299])
                    inception_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                    inception_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                    fake = (fake - inception_mean) / inception_std
                    # fake = torch.utils.data.TensorDataset(fake)
                    
                    inception_score = self.metric.inception_score(torch.utils.data.DataLoader(fake, batch_size=64))
                    fid = self.metric.fid(torch.utils.data.DataLoader(fake, batch_size=64))
                    
                    self.logger.debug(f'inception score: {inception_score}\tfid: {fid}')
                    
                    if low_fid > fid:
                        low_fid = fid
                        self.logger.debug(f"{iters} lowest fid")
                        self.save_model('fid', iters, low_fid, fake)
                    if high_is < inception_score:
                        high_is = inception_score
                        self.logger.debug(f"{iters} highest is")
                        self.save_model('is', iters, high_is, fake)
                    
                iters += 1

        end = time.time()
        self.logger.debug(f'train finish\ttiem: {time_to_str(end - start)}')
        
        plt.clf()
        epoch_x = range(1, len(losses_g) + 1)
        
        plt.figure(figsize=(8, 6))
        plt.plot(epoch_x, losses_d, label='Discriminator Loss')
        plt.plot(epoch_x, losses_g, label='Generator Loss')

        plt.title('Discriminator and Generator Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.model_save_path, f'loss.png'))
