import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
from torchinfo import summary
from torchvision import models
import torchvision.utils as vutils
import torchvision.transforms.functional as F

from preprocess.cgan_data_preprocessor import CGANDataPreprocessor
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
from model.CGAN import weights_init
from metrics import Metrics


class CGANTrainer(Trainer):
    def __init__(self, 
                 args: argparse.Namespace,
                 model_g: nn.Module,
                 model_d: nn.Module,
                 data_pre: CGANDataPreprocessor):
        self.logger = MainLogger(args)
        self.device = get_default_device()
        
        self.epoch = args.epoch
        self.max_lr = args.max_learning_rate
        
        self.model_g = model_g.to(self.device)
        self.model_d = model_d.to(self.device)
        
        self.lambda_gp = 10.0
        
        with torch.no_grad():
            self.logger.debug(f"Generator:\n{summary(self.model_g, input_size=[(1, 100, 1, 1), (1, 100)], dtypes=[torch.float, torch.long])}")
            self.logger.debug(f"Discriminator: {summary(self.model_d, input_size=[(1, 3, 64, 64), (1, 100)], dtypes=[torch.float, torch.long])}")
        self.model_g.apply(weights_init)
        self.model_d.apply(weights_init)

        self.data_pre = data_pre
        self.train_loader, metric_loader = self.data_pre.get_data_loader()
        self.metric = Metrics(metric_loader)
        
        self.optimizer_g = optim.Adam(self.model_g.parameters(), lr=self.max_lr, betas=[0.5, 0.999])
        self.optimizer_d = optim.Adam(self.model_d.parameters(), lr=self.max_lr, betas=[0.5, 0.999])
        
        self.criterion = nn.BCELoss()
        
        self.model_save_path = args.save_path
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        self.logger.debug(f'save path: {self.model_save_path}')
        
    
    def save_model(self, typ, iters, inception_score, fid, intra_fid, images):
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
        }, os.path.join(save_path, f'{iters}_{inception_score:.04f}_{fid:.04f}_{intra_fid:.04f}.pt'))
        
        self.save_image(save_path, iters, images)
        
        # self.logger.debug(f'{iters} model save')
        
    def save_image(self, path, iters, images):
        plt.clf()
        plt.axis("off")
        plt.title("fake images")
        plt.imshow(np.transpose(vutils.make_grid(images, padding=2, normalize=True, nrow=10),(1,2,0)))
        plt.savefig(os.path.join(path, f'{iters}_fake_image.png'))
        
    
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
        
        fixed_noise = []
        fixed_labels = []
        for i in range(100):
            noise = torch.randn(10, 100, 1, 1, device=self.device)
            labels_data = torch.LongTensor([1 if i == j else 0 for j in range(100)]).repeat(10, 1).to(self.device)
            
            fixed_noise.append(noise)
            fixed_labels.append(labels_data)
        fixed_noise = torch.vstack(fixed_noise)
        fixed_labels = torch.vstack(fixed_labels)
        
        low_fid = low_intra_fid = 1e10
        high_is = 0
        
        real_batch = next(iter(real_images_loader))
        plt.axis("off")
        plt.title("real images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(self.device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
        plt.savefig(os.path.join(self.model_save_path, 'real_image.png'))
        
        image_save_path = os.path.join(self.model_save_path, 'img')
        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)
        
        start = time.time()

        self.logger.debug("train start")
        for epoch in range(self.epoch):
            for i, data in enumerate(real_images_loader):
                self.model_d.zero_grad()
                
                real_data, labels_data = data
                real_data = real_data.to(self.device)
                labels_data = labels_data.to(self.device)
                
                b_size = real_data.size(0)
                label = torch.full((b_size,), label_real, dtype=torch.float32, device=self.device)
                real_data = 0.9 * real_data + 0.1 * torch.randn((real_data.size()), device=self.device)
                
                output = self.model_d(real_data, labels_data.detach()).view(-1)
                real_error_d = self.criterion(output, label)
                real_error_d.backward()
                x_d = output.mean().item()
                
                
                noise = torch.randn(b_size, 100, 1, 1, device=self.device)
                fake = self.model_g(noise, labels_data.detach())
                label.fill_(label_fake)
                fake = 0.9 * fake + 0.1 * torch.randn((fake.size()), device=self.device)
                
                output = self.model_d(fake.detach(), labels_data.detach()).view(-1)
                fake_error_d = self.criterion(output, label)
                fake_error_d.backward()
                z1_gd = output.mean().item()

                
                # gradient_penalty = self.compute_gradient_penalty(real_label_concat, fake_label_concat)
                # error_d = real_error_d + fake_error_d + self.lambda_gp * gradient_penalty
                error_d = real_error_d + fake_error_d
                # gradient_penalty.backward()
                self.optimizer_d.step()

                self.model_g.zero_grad()
                label.fill_(label_real)
                
                output = self.model_d(fake, labels_data).view(-1)
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
                        generated_fake = self.model_g(fixed_noise, fixed_labels).detach().cpu()
                    
                    generated_fake = 0.5 * generated_fake + 0.5
                    generated_fake = F.resize(generated_fake, [299, 299])
                    inception_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                    inception_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                    generated_fake = (generated_fake - inception_mean) / inception_std
                    # fake = torch.utils.data.TensorDataset(fake)
                    
                    inception_score = self.metric.inception_score(torch.utils.data.DataLoader(generated_fake, batch_size=128))
                    fid = self.metric.fid(torch.utils.data.DataLoader(generated_fake, batch_size=128))
                    intra_fid = self.metric.intra_fid(generated_fake)
                    
                    self.logger.debug(f'inception score: {inception_score}\tfid: {fid}\tintra fid: {intra_fid}')
                    
                    if low_fid > fid:
                        low_fid = fid
                        self.logger.debug(f"{iters} lowest fid")
                        self.save_model('fid', iters, inception_score, fid, intra_fid, generated_fake[::10])
                    if low_intra_fid > intra_fid:
                        low_intra_fid = intra_fid
                        self.logger.debug(f"{iters} lowest intra fid")
                        self.save_model('intra_fid', iters, inception_score, fid, intra_fid, generated_fake[::10])
                    if high_is < inception_score:
                        high_is = inception_score
                        self.logger.debug(f"{iters} highest is")
                        self.save_model('is', iters, inception_score, fid, intra_fid, generated_fake[::10])
                        
                    self.save_image(image_save_path, iters, generated_fake[::10])
                    
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
