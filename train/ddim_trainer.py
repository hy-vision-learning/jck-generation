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
        
        
    def infiniteloop(self, dataloader):
        while True:
            for x, y in iter(dataloader):
                yield x, y
        
    
    def __load_data(self):
        dataset = CIFAR100("./data", train=True, download=True, 
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
        inceptionset = CIFAR100("./data", train=True, download=True,
            transform=transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        )
        
        loader = DataLoader(dataset, batch_size=self.args.batch_size,
                            num_workers=self.args.num_workers, shuffle=True, pin_memory=True)
        return dataset, loader, inceptionset
    
    
    def __save_model(self, typ, epoch, model_name, model, model_ema, optimizer, fixed_noise):
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
            'optimizer': optimizer.state_dict(),
            'fixed_noise': fixed_noise
        }, os.path.join(save_path, model_name))
        
        
    def __load_model(self):
        self.logger.debug(f'Loading model from {os.getcwd() + self.args.load_model}')
        model_path = self.args.load_model
        data = torch.load(model_path)
        return data['epoch'], data['model'], data['model_ema'], data['optimizer'], data['fixed_noise']
        
    
    def __evaluate(self, metric, noise_loader, steps, file_name=None, method="linear", eta=0.0, only_return_x_0=True):
        self.ema_helper.ema(self.ema_model)
        with torch.no_grad():
            generated_images = []
            for fixed_noise in tqdm(noise_loader, desc="Generating images", ncols=0):
                fixed_noise = fixed_noise.to(self.device)
                generated_image = self.sampler(fixed_noise, steps=steps, method=method, eta=eta, only_return_x_0=only_return_x_0)
                generated_images.append(generated_image.cpu())
            generated_images = torch.cat(generated_images, dim=0)
        
        if file_name is not None:
            sample_image = torch.clamp(generated_images, -1.0, 1.0)
            grid = (make_grid(sample_image)[:64] + 1) / 2
            path = os.path.join(self.args.save_path, 'sample')
            if not os.path.exists(path):
                os.makedirs(path)
            image_path = os.path.join(path, file_name)
            save_image(grid, image_path)
        
        images = torch.clamp(generated_images, 0.0, 1.0)
        result = TF.resize(images, [299, 299])
        inception_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        inception_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        result = (result - inception_mean) / inception_std
        
        inception_score = metric.inception_score(torch.utils.data.DataLoader(result, batch_size=64))
        fid = metric.fid(torch.utils.data.DataLoader(result, batch_size=64))
        
        self.logger.debug(f'IS: {inception_score[0]:.4f}\tFID: {fid:.4f}')
        return inception_score[0], fid
    
    
    def train(self):
        dataset, dataloader, inceptionset = self.__load_data()
        metric = Metrics(inceptionset)
        
        datalooper = self.infiniteloop(dataloader)
        
        self.unet_model = UNet(dropout=self.args.dropout).to(self.device)
        forward_trainer = DDIMForwardTrainer(self.unet_model, self.args.beta_1, self.args.beta_T, self.args.T).to(self.device)
        
        self.ema_helper = EMAHelper(mu=0.999, device=self.device)
        self.ema_helper.register(self.unet_model)
        self.ema_model = self.ema_helper.ema_copy(self.unet_model)
        
        optimizer = torch.optim.Adam(self.unet_model.parameters(), lr=self.args.lr, 
                                      weight_decay=self.args.weight_decay, betas=(0.9, 0.999))
        
        start_step = 0
        fixed_noise_origin = torch.randn((self.args.sample_size, 3, 32, 32))
        
        if self.args.load_model:
            start_step, model_state, model_ema_state, optimizer_state, _ =  self.__load_model()
            self.unet_model.load_state_dict(model_state)
            self.ema_model.load_state_dict(model_ema_state)
            optimizer.load_state_dict(optimizer_state)
            
            self.ema_helper.register(self.ema_model)
        
        self.sampler = DDIMSampler(self.ema_model, self.args.beta_1, self.args.beta_T, self.args.T).to(self.device)
        
        fixed_noise_loader = DataLoader(fixed_noise_origin, batch_size=self.args.batch_size * 2,
                            num_workers=0, pin_memory=True)
        
        highest_is = 0
        lowest_fid = 1e10
        
        start_time = time.time()
        step_time = time.time()
        for step in range(start_step, self.args.total_steps):
            images, _ = next(datalooper)
            x_0 = images.to(self.device)
                
            loss = forward_trainer(x_0)
            
            optimizer.zero_grad()
            loss.backward()
            
            if step % 50 == 0 or step == self.args.total_steps - 1:
                self.logger.debug(
                    f"step: {step}\tloss: {loss.item():.4f}\ttime: {time_to_str(time.time() - step_time)}"
                )
                step_time = time.time()
            
            if self.args.grad_clip != -1:
                torch.nn.utils.clip_grad_norm_(self.unet_model.parameters(), self.args.grad_clip)
            
            optimizer.step()
            self.ema_helper.update(self.unet_model)
            
            if step % self.args.eval_step == 0 or step == self.args.total_steps - 1:
                inception_score, fid = self.__evaluate(metric, fixed_noise_loader, file_name=f'{step}.png',
                                                       steps=self.args.eval_sample_step, method="linear",
                                                       eta=0.0, only_return_x_0=True)
                
                model_name = f'{step}_{inception_score:.04f}_{fid:.04f}.pt'
                if inception_score > highest_is:
                    highest_is = inception_score
                    self.__save_model('is', step, model_name, self.unet_model, self.ema_model, optimizer, fixed_noise_origin)
                    self.logger.debug(f'highest IS')
                if fid < lowest_fid:
                    lowest_fid = fid
                    self.__save_model('fid', step, model_name, self.unet_model, self.ema_model, optimizer, fixed_noise_origin)
                    self.logger.debug(f'lowest FID')
                
            self.__save_model('model', step, 'last.pt', self.unet_model, self.ema_model, optimizer, fixed_noise_origin)
        self.logger.debug(f"training time: {time_to_str(time.time() - start_time)}")
        
        inception_score, fid = self.__evaluate(metric, fixed_noise_loader, file_name=f'final.png',
                                                       steps=self.args.sample_step, method="linear",
                                                       eta=0.0, only_return_x_0=True)


    def test(self):
        _, _, inceptionset = self.__load_data()
        metric = Metrics(inceptionset)
        
        self.unet_model = UNet(dropout=self.args.dropout).to(self.device)
        
        self.ema_helper = EMAHelper(mu=0.999, device=self.device)
        self.ema_helper.register(self.unet_model)
        self.ema_model = self.ema_helper.ema_copy(self.unet_model)
        
        optimizer = torch.optim.Adam(self.unet_model.parameters(), lr=self.args.lr, 
                                      weight_decay=self.args.weight_decay, betas=(0.9, 0.999))
        
        fixed_noise = torch.randn((self.args.sample_size, 3, 32, 32))
        
        if self.args.load_model:
            _, model_state, model_ema_state, optimizer_state, _ =  self.__load_model()
            self.unet_model.load_state_dict(model_state)
            self.ema_model.load_state_dict(model_ema_state)
            optimizer.load_state_dict(optimizer_state)
            
            self.ema_helper.register(self.ema_model)
        
        self.sampler = DDIMSampler(self.ema_model, self.args.beta_1, self.args.beta_T, self.args.T).to(self.device)
        
        fixed_noise_loader = DataLoader(fixed_noise, batch_size=self.args.batch_size * 2,
                            num_workers=0, pin_memory=True)
        
        self.__evaluate(metric, fixed_noise_loader, file_name=f'sample.png',
                            steps=self.args.sample_step, method="linear",
                            eta=0.0, only_return_x_0=True)
