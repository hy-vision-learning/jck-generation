import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import CIFAR100
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid, save_image

from model.DDIM.unet import UNet
# from model.DDIM.diffusion import DDIMForwardTrainer, DDIMSampler, EMAHelper
from model.BIGGAN.BIGGAN import Generator, Discriminator, BigGAN

from metrics import Metrics
from logger.main_logger import MainLogger
from logger.utils import time_to_str

from utils import get_default_device
from tqdm import tqdm, trange
import copy
import os
import json
import time

import functools

import os
import functools
import math
import numpy as np
from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision

import model.BIGGAN.BIGGAN as model
from datetime import datetime

import inceptionID


class EMA(object):
    def __init__(self, source, target, decay=0.9999, start_itr=0):
        self.source = source
        self.target = target
        self.decay = decay
        
        self.start_itr = start_itr
        
        self.source_dict = self.source.state_dict()
        self.target_dict = self.target.state_dict()
        
        with torch.no_grad():
            for key in self.source_dict:
                self.target_dict[key].data.copy_(self.source_dict[key].data)

    def update(self, itr=None):
        if itr and itr < self.start_itr:
            decay = 0.0
        else:
            decay = self.decay
        with torch.no_grad():
            for key in self.source_dict:
                self.target_dict[key].data.copy_(self.target_dict[key].data * decay 
                                                + self.source_dict[key].data * (1 - decay))
                
                
class Distribution(torch.Tensor):
    def init_distribution(self, dist_type, **kwargs):    
        self.dist_type = dist_type
        self.dist_kwargs = kwargs
        if self.dist_type == 'normal':
            self.mean, self.var = kwargs['mean'], kwargs['var']
        elif self.dist_type == 'categorical':
            self.num_categories = kwargs['num_categories']

    def sample_(self):
        if self.dist_type == 'normal':
            self.normal_(self.mean, self.var)
        elif self.dist_type == 'categorical':
            self.random_(0, self.num_categories)
    
    
    def to(self, *args, **kwargs):
        new_obj = Distribution(self)
        new_obj.init_distribution(self.dist_type, **self.dist_kwargs)
        new_obj.data = super().to(*args, **kwargs)    
        return new_obj


class BIGGANTrainer:
    def __init__(self, args):
        self.args = args
        self.device = get_default_device()
        self.logger = MainLogger(self.args)
        self.metrics = Metrics(self.args)
        self.args.num_classes = 20 if self.args.superclass else 100
        self.superclass_mapping = inceptionID.super_class_mapping()
        
    
    def get_data_loader(self, batch_size):
        train_set = CIFAR100(root='./data', train=True, download=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                    ]))
        
        if self.args.superclass:
            train_set.targets = [self.superclass_mapping[label] for label in train_set.targets]
        
        train_loader = DataLoader(train_set, batch_size=batch_size,  shuffle=True,
                                  drop_last=True, pin_memory=True, num_workers=self.args.num_workers)
        return train_loader
    
    
    def prepare_z_y(self, G_batch_size, dim_z, nclasses,z_var=1.0):
        z_ = Distribution(torch.randn(G_batch_size, dim_z, requires_grad=False))
        z_.init_distribution('normal', mean=0, var=z_var)
        z_ = z_.to(self.device, torch.float32)

        y_ = Distribution(torch.zeros(G_batch_size, requires_grad=False))
        y_.init_distribution('categorical',num_categories=nclasses)
        y_ = y_.to(self.device, torch.int64)
        return z_, y_
    
    
    def sample(self, G, z_, y_, set_labels=False, labels=None):
        with torch.no_grad():
            z_.sample_()
            
            if set_labels:
                y_.copy_(torch.tensor(labels))
            else:
                y_.sample_()
                
            G_z = G(z_, G.shared(y_))
            return G_z, y_
        
        
    def ortho(self, model, strength=1e-4, blacklist=[]):
        with torch.no_grad():
            for param in model.parameters():
                # Only apply this to parameters with at least 2 axes, and not in the blacklist
                if len(param.shape) < 2 or any([param is item for item in blacklist]):
                    continue
                w = param.view(param.shape[0], -1)
                grad = (2 * torch.mm(torch.mm(w, w.t()) 
                        * (1. - torch.eye(w.shape[0], device=w.device)), w))
                param.grad.data += strength * grad.view(param.shape)
    
    
    def loss_hinge_dis(self, dis_fake, dis_real):
        loss_real = torch.mean(F.relu(1. - dis_real))
        loss_fake = torch.mean(F.relu(1. + dis_fake))
        return loss_real, loss_fake


    def loss_hinge_gen(self, dis_fake):
        loss = -torch.mean(dis_fake)
        return loss
    
    
    def __save_model(self, typ, epoch, model_name):
        save_path = os.path.join(self.args.save_path, typ)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        for filename in os.listdir(save_path):
            file_path = os.path.join(save_path, filename)
            if os.path.isfile(file_path) and filename.endswith('.pt'):
                os.remove(file_path)
        
        torch.save({
            'epoch': epoch,
            'model_g': self.model_g.state_dict(),
            'model_d': self.model_d.state_dict(),
            'optim_g': self.optim_g.state_dict(),
            'optim_d': self.optim_d.state_dict(),
            'model_ema': self.ema_g.state_dict(),
            'state_dict': self.state_dict
        }, os.path.join(save_path, model_name))
    
    
    def train(self, x, y, z_, y_):
        def toggle_grad(model, on_or_off):
            for param in model.parameters():
                param.requires_grad = on_or_off
        
        self.optim_g.zero_grad()
        self.optim_d.zero_grad()
        
        x = torch.split(x, self.args.batch_size)
        y = torch.split(y, self.args.batch_size)
        counter = 0
        
        toggle_grad(self.model_d, True)
        toggle_grad(self.model_g, False)
        
        for step_index in range(self.args.num_D_steps):
            self.optim_d.zero_grad()
            
            z_.sample_()
            y_.sample_()
            D_fake, D_real = self.biggan(z_[:self.args.batch_size], y_[:self.args.batch_size], 
                                x[counter], y[counter], train_G=False)
            
            D_loss_real, D_loss_fake = self.loss_hinge_dis(D_fake, D_real)
            D_loss = (D_loss_real + D_loss_fake)
            D_loss.backward()
            counter += 1
            
            if self.args.D_ortho > 0.0:
                self.ortho(self.model_d, self.args.D_ortho)
        
            self.optim_d.step()
        
        toggle_grad(self.model_d, False)
        toggle_grad(self.model_g, True)
        
        self.optim_g.zero_grad()
          
        z_.sample_()
        y_.sample_()
        D_fake = self.biggan(z_, y_, train_G=True)
        G_loss = self.loss_hinge_gen(D_fake)
        G_loss.backward()
        
        if self.args.G_ortho > 0.0:
            self.ortho(self.model_g, self.args.G_ortho, 
                        blacklist=[param for param in self.model_g.shared.parameters()])
        self.optim_g.step()
        
        self.ema.update(self.state_dict['itr'])
        
        return float(G_loss.item()), float(D_loss_real.item()), float(D_loss_fake.item())
    
    
    def test(self, sample, full=False):
        self.logger.debug(f'full test: {full}')
        IS_mean, IS_std, FID, intra_FID = self.metrics.get_inception_metrics(sample, self.args.num_inception_images, num_splits=10, full=full, superclass=self.args.superclass)
        if not full:
            self.logger.debug(f'Saved metrics: IS: {self.state_dict["best_IS"]}, FID: {self.state_dict["best_FID"]}, intra-FID: {self.state_dict["best_intra_FID"]}')
            return
        
        if not math.isnan(IS_mean) and self.state_dict['best_IS'] < IS_mean:
            self.logger.debug(f'best IS: {self.state_dict['best_IS']} -> {IS_mean}')
            self.state_dict['best_IS'] = IS_mean
            self.__save_model('is', self.state_dict['itr'], 'best.pt')
        if not math.isnan(FID) and self.state_dict['best_FID'] > FID:
            self.logger.debug(f'best FID: {self.state_dict['best_FID']} -> {FID}')
            self.state_dict['best_FID'] = FID
            self.__save_model('FID', self.state_dict['itr'], 'best.pt')
        if not math.isnan(intra_FID) and self.state_dict['best_intra_FID'] > intra_FID:
            self.logger.debug(f'best intra-FID: {self.state_dict['best_intra_FID']} -> {intra_FID}')
            self.state_dict['best_intra_FID'] = intra_FID
            self.__save_model('intra_FID', self.state_dict['itr'], 'best.pt')
        
        
    def save_and_sample(self, z_, y_, fixed_z, fixed_y):
        self.__save_model('sample', self.state_dict['epoch'], 'sample.pt')
           
        with torch.no_grad():
            fixed_Gz = self.ema_g(fixed_z, self.ema_g.shared(fixed_y))
        
        fixed_Gz = torch.tensor(fixed_Gz.tolist())
        image_path = os.path.join(self.args.save_path, 'sample', 'image', f'{self.state_dict["itr"]}')
        if not os.path.isdir(image_path):
            os.makedirs(image_path)
        image_filename = os.path.join(image_path, f'fixed_samples{self.state_dict["itr"]}.jpg')
        torchvision.utils.save_image(fixed_Gz.detach().float().cpu(), image_filename, nrow=int(fixed_Gz.shape[0] ** 0.5), normalize=True)
        
        ims = []
        y = torch.arange(0, self.args.num_classes, device='cuda')
        for j in range(10):
            z_ = torch.randn(self.args.num_classes, self.model_g.dim_z, device='cuda')        
            with torch.no_grad():
                o = self.model_g(z_, self.model_g.shared(y))

            ims += [o.data.cpu()]
        
        all_images = torch.stack(ims, 1).view(-1, ims[0].shape[1], ims[0].shape[2], ims[0].shape[3]).data.float().cpu()
        image_filename = os.path.join(image_path, f'samples{self.state_dict["itr"]}.jpg')
        torchvision.utils.save_image(all_images, image_filename, nrow=10, normalize=True)
        
    def run(self):
        self.model_g = model.Generator().to(self.device)
        self.model_d = model.Discriminator().to(self.device)
        
        self.optim_g = optim.Adam(params=self.model_g.parameters(), lr=2e-4, betas=(0.0, 0.999))
        self.optim_d = optim.Adam(params=self.model_d.parameters(), lr=2e-4, betas=(0.0, 0.999))
        
        self.ema_g = model.Generator(skip_init=True).to(self.device)
        self.ema = EMA(self.model_g, self.ema_g, self.args.ema_decay, self.args.ema_start)
        
        self.biggan = model.BigGAN(self.model_g, self.model_d)
        # self.logger.debug(f'G: {G}\nD: {D}')
        
        self.logger.debug('Number of params in G: {} D: {}'.format(
            *[sum([p.data.nelement() for p in net.parameters()]) for net in [self.model_g, self.model_d]]))
        
        self.state_dict = {'itr': 0, 'epoch': 0, 'best_IS': 0, 'best_FID': 999999, 'best_intra_FID': 999999}
        
        D_batch_size = self.args.batch_size * self.args.num_D_steps
        loader = self.get_data_loader(batch_size=D_batch_size)

        z_, y_ = self.prepare_z_y(self.args.batch_size, self.model_g.dim_z, self.args.num_classes)
        fixed_z, fixed_y = self.prepare_z_y(self.args.batch_size, self.model_g.dim_z, self.args.num_classes)  
        
        fixed_z.sample_()
        fixed_y.sample_()
        
        sample = functools.partial(self.sample, G=self.ema_g, z_=z_, y_=y_)
        
        full_test_counter = 1
        for epoch in range(self.state_dict['epoch'], self.args.num_epochs):
            for i, (x, y) in enumerate(tqdm(loader, ncols=0)):
                self.state_dict['itr'] += 1
                
                self.model_g.train()
                self.model_d.train()
                self.ema_g.train()
                
                x, y = x.to(self.device), y.to(self.device)
                G_loss, D_loss_real, D_loss_fake = self.train(x, y, z_, y_)
                
                if not (self.state_dict['itr'] % 100): 
                    self.logger.debug(f'itr: {self.state_dict["itr"]}\tG_loss: {G_loss:.4f}\t' + 
                                      f'D_loss_real: {D_loss_real:.4f}\tD_loss_fake: {D_loss_fake:.4f}')
                
                if not (self.state_dict['itr'] % self.args.save_every):
                    self.save_and_sample(z_, y_, fixed_z, fixed_y)

                if not (self.state_dict['itr'] % self.args.test_every):
                    if self.args.full_test_counter and not (full_test_counter % self.args.full_test_counter):
                        self.test(sample, full=True)
                        full_test_counter = 1
                    else:
                        self.test(sample)
                        full_test_counter += 1
                        self.logger.debug(f'Full test remains: {self.args.full_test_counter - full_test_counter}')
            
            self.state_dict['epoch'] += 1
