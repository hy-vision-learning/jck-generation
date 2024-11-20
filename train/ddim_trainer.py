import torch
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import CIFAR100
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid, save_image

from model.DDIM.unet import UNet
from model.DDIM.diffusion import DDIMForwardTrainer, DDIMSampler, EMAHelper

from metrics import Metrics
from logger.main_logger import MainLogger
from logger.utils import time_to_str

from utils import get_default_device
from tqdm import tqdm, trange
import copy
import os
import json
import time


class DDIMTrainer:
    def __init__(self, args):
        self.args = args
        self.device = get_default_device()
        self.logger = MainLogger(self.args)
        
    
    def __load_data(self):
        dataset = CIFAR100("./data", train=True, download=True, 
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
        loader = DataLoader(dataset, batch_size=self.args.batch_size,
                            num_workers=self.args.num_workers, shuffle=True, pin_memory=True)
        # datalooper = self.infiniteloop(loader)
        return dataset, loader
    
    
    def __save_model(self, typ, epoch, model_name, model, model_ema, optimizer):
        save_path = os.path.join(self.args.save_path, typ)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        for filename in os.listdir(save_path):
            file_path = os.path.join(save_path, filename)
            if os.path.isfile(file_path) and filename.endswith('.pt'):
                os.remove(file_path)
        
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'model_ema': model_ema.state_dict(),
            'optimizer': optimizer.state_dict()
        }, os.path.join(save_path, model_name))
        
    
    def __evaluate(self, metric, images):
        result = TF.resize(images, [299, 299])
        inception_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        inception_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        result = (result - inception_mean) / inception_std
        
        inception_score = metric.inception_score(torch.utils.data.DataLoader(result, batch_size=64))
        fid = metric.fid(torch.utils.data.DataLoader(result, batch_size=64))
        return inception_score, fid
    
    
    def train(self):
        dataset, datalooper = self.__load_data()
        metric = Metrics(dataset)
        
        unet_model = UNet(dropout=self.args.dropout).to(self.device)
        forward_trainer = DDIMForwardTrainer(unet_model, self.args.beta_1, self.args.beta_T, self.args.T).to(self.device)
        
        ema_helper = EMAHelper(mu=0.999, device=self.device)
        ema_helper.register(unet_model)
        ema_model = ema_helper.ema_copy(unet_model)
        
        sampler = DDIMSampler(ema_model, self.args.beta_1, self.args.beta_T, self.args.T).to(self.device)
        
        optimizer = torch.optim.Adam(unet_model.parameters(), lr=self.args.lr, 
                                      weight_decay=self.args.weight_decay, betas=(0.9, 0.999))
        
        start_epoch = step = 0
        start_time = time.time()
        
        fixed_noise = torch.randn((self.args.sample_size, 3, 32, 32))
        fixed_noise_loader = DataLoader(fixed_noise, batch_size=self.args.batch_size * 2,
                            num_workers=0, pin_memory=True)
        
        highest_is = 0
        lowest_fid = 1e10
        
        for epoch in range(start_epoch, self.args.total_steps + 1):
            step_time = time.time()
            epoch_loss = 0
            for i, (images, _) in enumerate(datalooper):
                x_0 = images.to(self.device)
                
                loss = forward_trainer(x_0)
                
                optimizer.zero_grad()
                loss.backward()
                
                epoch_loss += loss.item()
                if i % 50 == 0 or i == len(datalooper) - 1:
                    self.logger.debug(
                        f"epoch: {epoch}\tstep: {i}\tloss: {loss.item():.4f}"
                    )   
                
                if self.args.grad_clip != -1:
                    torch.nn.utils.clip_grad_norm_(unet_model.parameters(), self.args.grad_clip)
                
                optimizer.step()
                ema_helper.update(unet_model)
                
            if epoch % self.args.eval_step == 0 or epoch == self.args.total_steps - 1:
                ema_helper.ema(ema_model)
                with torch.no_grad():
                    generated_images = []
                    for fixed_noise in tqdm(fixed_noise_loader, desc="Generating images"):
                        fixed_noise = fixed_noise.to(self.device)
                        generated_image = sampler(fixed_noise, steps=self.args.eval_sample_step, method="linear", eta=0.0, only_return_x_0=True)
                        generated_images.append(generated_image.cpu())
                    generated_images = torch.cat(generated_images, dim=0)
                generated_images = torch.clamp(generated_images, -1.0, 1.0)
                
                inception_score, fid = self.__evaluate(metric, generated_images)
                self.logger.debug(f'IS: {inception_score:.4f}\tFID: {fid:.4f}')
                
                model_name = f'{epoch}_{inception_score:.04f}_{fid:.04f}.pt'
                if inception_score > highest_is:
                    highest_is = inception_score
                    self.__save_model('is', epoch, model_name, unet_model, ema_model, optimizer)
                    self.logger.debug(f'highest IS')
                if fid < lowest_fid:
                    lowest_fid = fid
                    self.__save_model('fid', epoch, model_name, unet_model, ema_model, optimizer)
                    self.logger.debug(f'lowest FID')
                
                grid = (make_grid(generated_images)[:64] + 1) / 2
                path = os.path.join(self.args.save_path, 'sample')
                if not os.path.exists(path):
                    os.makedirs(path)
                image_path = os.path.join(path, '%d.png' % epoch)
                save_image(grid, image_path)
                
            self.__save_model('model', epoch, 'last.pt', unet_model, ema_model, optimizer)
            self.logger.debug(f"Epoch {epoch}/{self.args.total_steps}" +
                f"\ttime: {time_to_str(time.time() - step_time)}\tloss: {epoch_loss / len(datalooper):.4f}")
        
        self.logger.debug(f"training time: {time_to_str(time.time() - start_time)}")
        
        ema_helper.ema(ema_model)
        with torch.no_grad():
            generated_images = []
            for fixed_noise in tqdm(fixed_noise_loader, desc="Generating images"):
                fixed_noise = fixed_noise.to(self.device)
                generated_image = sampler(fixed_noise, steps=self.args.eval_sample_step, method="linear", eta=0.0, only_return_x_0=True)
                generated_images.append(generated_image.cpu())
            generated_images = torch.cat(generated_images, dim=0)
        generated_images = torch.clamp(generated_images, -1.0, 1.0)
        inception_score, fid = self.__evaluate(metric, generated_images)
        self.logger.debug(f'Final Results\tIS: {inception_score:.4f}\tFID: {fid:.4f}')
