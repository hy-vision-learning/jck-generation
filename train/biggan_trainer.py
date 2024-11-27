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
from model.BIGGAN.BIGGAN import Generator, Discriminator, G_D

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

# Import my stuff
from model.BIGGAN.sync_batchnorm import patch_replication_callback

import model.BIGGAN.BIGGAN as model
from datetime import datetime


class EMA(object):
    def __init__(self, source, target, decay=0.9999, start_itr=0):
        self.source = source
        self.target = target
        self.decay = decay
        # Optional parameter indicating what iteration to start the decay at
        self.start_itr = start_itr
        # Initialize target's params to be source's
        self.source_dict = self.source.state_dict()
        self.target_dict = self.target.state_dict()
        print('Initializing EMA parameters to be source parameters...')
        with torch.no_grad():
            for key in self.source_dict:
                self.target_dict[key].data.copy_(self.source_dict[key].data)
            # target_dict[key].data = source_dict[key].data # Doesn't work!

    def update(self, itr=None):
        # If an iteration counter is provided and itr is less than the start itr,
        # peg the ema weights to the underlying weights.
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
        
    
    def get_data_loader(self, batch_size):
        train_set = CIFAR100(root='./data', train=True, download=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                    ]))
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
    
    
    def __save_model(self, typ, epoch, model_name, G, D, state_dict, G_ema):
        save_path = os.path.join(self.args.save_path, typ)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        for filename in os.listdir(save_path):
            file_path = os.path.join(save_path, filename)
            if os.path.isfile(file_path) and filename.endswith('.pt'):
                os.remove(file_path)
        
        torch.save({
            'epoch': epoch,
            'model_g': G.state_dict(),
            'model_d': D.state_dict(),
            'optim_g': G.optim.state_dict(),
            'optim_d': D.optim.state_dict(),
            'model_ema': G_ema.state_dict(),
            'state_dict': state_dict
        }, os.path.join(save_path, model_name))
    
    
    def train(self, x, y, G, D, GD, z_, y_, ema, state_dict):
        def toggle_grad(model, on_or_off):
            for param in model.parameters():
                param.requires_grad = on_or_off
        
        G.optim.zero_grad()
        D.optim.zero_grad()
        # How many chunks to split x and y into?
        x = torch.split(x, self.args.batch_size)
        y = torch.split(y, self.args.batch_size)
        counter = 0
        
        toggle_grad(D, True)
        toggle_grad(G, False)
        
        for step_index in range(self.args.num_D_steps):
        # If accumulating gradients, loop multiple times before an optimizer step
            D.optim.zero_grad()
            for accumulation_index in range(self.args.num_D_accumulations):
                z_.sample_()
                y_.sample_()
                D_fake, D_real = GD(z_[:self.args.batch_size], y_[:self.args.batch_size], 
                                    x[counter], y[counter], train_G=False)
                
                # Compute components of D's loss, average them, and divide by 
                # the number of gradient accumulations
                D_loss_real, D_loss_fake = self.loss_hinge_dis(D_fake, D_real)
                D_loss = (D_loss_real + D_loss_fake) / float(self.args.num_D_accumulations)
                D_loss.backward()
                counter += 1
            
            # Optionally apply ortho reg in D
            if self.args.D_ortho > 0.0:
                # Debug print to indicate we're using ortho reg in D.
                print('using modified ortho reg in D')
                self.ortho(D, self.args.D_ortho)
        
            D.optim.step()
        
        toggle_grad(D, False)
        toggle_grad(G, True)
        
        # Zero G's gradients by default before training G, for safety
        G.optim.zero_grad()
        
        # If accumulating gradients, loop multiple times
        for accumulation_index in range(self.args.num_G_accumulations):    
            z_.sample_()
            y_.sample_()
            D_fake = GD(z_, y_, train_G=True)
            G_loss = self.loss_hinge_gen(D_fake) / float(self.args.num_G_accumulations)
            G_loss.backward()
        
        # Optionally apply modified ortho reg in G
        if self.args.G_ortho > 0.0:
            print('using modified ortho reg in G') # Debug print to indicate we're using ortho reg in G
            # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
            self.ortho(G, self.args.G_ortho, 
                        blacklist=[param for param in G.shared.parameters()])
        G.optim.step()
            
        # If we have an ema, update it, regardless of if we test with it or not
        ema.update(state_dict['itr'])
        
        out = {'G_loss': float(G_loss.item()), 
                'D_loss_real': float(D_loss_real.item()),
                'D_loss_fake': float(D_loss_fake.item())}
        # Return G's loss and the components of D's loss.
        return out
    
    
    def test(self, G, D, G_ema, state_dict, sample, get_inception_metrics):
        print('Gathering inception metrics...')
        IS_mean, IS_std, FID, intra_FID = get_inception_metrics(sample, self.args.num_inception_images, num_splits=10)
        # IS_mean, IS_std, FID = get_inception_metrics(sample, config['num_inception_images'], num_splits=10)
        # print('Itr %d: PYTORCH UNOFFICIAL Inception Score is %3.3f +/- %3.3f, PYTORCH UNOFFICIAL FID is %5.4f' % (state_dict['itr'], IS_mean, IS_std, FID))
        
        if ((self.args.which_best == 'IS' and IS_mean > state_dict['best_IS'])
            or (self.args.which_best == 'FID' and FID < state_dict['best_FID'])):
            print('%s improved over previous best, saving checkpoint...' % self.args.which_best)
            self.__save_model('best', state_dict['epoch'], 'best.pt', G, D, state_dict, G_ema)
            state_dict['save_best_num'] = (state_dict['save_best_num'] + 1 ) % self.args.num_best_copies
        state_dict['best_IS'] = max(state_dict['best_IS'], IS_mean)
        state_dict['best_FID'] = min(state_dict['best_FID'], FID)
        
        
    def save_and_sample(self, G, D, G_ema, z_, y_, fixed_z, fixed_y, state_dict):
        self.__save_model('sample', state_dict['epoch'], 'sample.pt', G, D, state_dict, G_ema)
           
        with torch.no_grad():
            fixed_Gz = G_ema(fixed_z, G_ema.shared(fixed_y))
        
        fixed_Gz = torch.tensor(fixed_Gz.tolist())
        image_path = os.path.join(self.args.save_path, 'sample', 'image')
        if not os.path.isdir(image_path):
            os.mkdir(image_path)
        image_filename = os.path.join(image_path, f'fixed_samples{state_dict["itr"]}.jpg')
        torchvision.utils.save_image(fixed_Gz.detach().float().cpu(), image_filename, nrow=int(fixed_Gz.shape[0] **0.5), normalize=True)
        
        
    def run(self):
        self.args.weights_root = f'{self.args.save_path}/weights'
        self.args.samples_root = f'{self.args.save_path}/samples'
        self.args.data_root = './data'
        if not os.path.exists(self.args.weights_root):
            os.mkdir(self.args.weights_root)
        if not os.path.exists(self.args.samples_root):
            os.mkdir(self.args.samples_root)

        G = model.Generator().to(self.device)
        D = model.Discriminator().to(self.device)
        
        G_ema = model.Generator(skip_init=True, no_optim=True).to(self.device)
        ema = EMA(G, G_ema, self.args.ema_decay, self.args.ema_start)
        
        GD = model.G_D(G, D)
        self.logger.debug(f'G: {G}\nD: {D}')
        
        self.logger.debug('Number of params in G: {} D: {}'.format(
            *[sum([p.data.nelement() for p in net.parameters()]) for net in [G,D]]))
        
        state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                        'best_IS': 0, 'best_FID': 999999}
        
        D_batch_size = self.args.batch_size * self.args.num_D_steps * self.args.num_D_accumulations
        loader = self.get_data_loader(batch_size=D_batch_size)

        z_, y_ = self.prepare_z_y(self.args.batch_size, G.dim_z, 100)
        fixed_z, fixed_y = self.prepare_z_y(self.args.batch_size, G.dim_z, 100)  
        
        fixed_z.sample_()
        fixed_y.sample_()
        
        sample = functools.partial(self.sample, G=G_ema, z_=z_, y_=y_)

        print('Beginning training at epoch %d...' % state_dict['epoch'])
        
        for epoch in range(state_dict['epoch'], self.args.num_epochs):
            for i, (x, y) in enumerate(tqdm(loader, ncols=0)):
                state_dict['itr'] += 1
                
                G.train()
                D.train()
                G_ema.train()
                
                x, y = x.to(self.device), y.to(self.device)
                metrics = self.train(x, y, G, D, GD, z_, y_, ema, state_dict)
                # metrics = train(x, y)
                
                if not (state_dict['itr'] % 100): 
                    self.logger.debug(f'itr: {state_dict["itr"]}\t{"\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()])}')
                
                if not (state_dict['itr'] % self.args.save_every):
                    # G.eval()
                    # G_ema.eval()
                    self.save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y, state_dict)

                if not (state_dict['itr'] % self.args.test_every):
                    # G.eval()
                    self.test(G, D, G_ema, state_dict, sample,
                                self.metrics.get_inception_metrics)
            # Increment epoch counter at end of epoch
            state_dict['epoch'] += 1
