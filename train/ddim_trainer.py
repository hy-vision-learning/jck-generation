import torch
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import CIFAR100
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid, save_image

from model.DDIM.unet import UNet
from model.DDIM.diffusion import GaussianDiffusionTrainer, DDIMSampler

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
    
    def ema(self, source, target, decay):
        source_dict = source.state_dict()
        target_dict = target.state_dict()
        for key in source_dict.keys():
            target_dict[key].data.copy_(
                target_dict[key].data * decay +
                source_dict[key].data * (1 - decay))


    def infiniteloop(self, dataloader):
        while True:
            for x, y in iter(dataloader):
                yield x, y
                
    def evaluate(self, args, sampler, model, device, metric: Metrics):
        model.eval()
        with torch.no_grad():
            images = []
            desc = "generating images"
            for i in trange(0, args.num_images, args.batch_size, desc=desc):
                batch_size = min(args.batch_size, args.num_images - i)
                x_T = torch.randn((batch_size, 3, args.img_size, args.img_size), device=device)
                batch_images = sampler(x_T.to(device))
                images.append((batch_images + 1) / 2)
            images = torch.cat(images, dim=0)
        model.train()
        
        # print(images[0])
        
        # images = 0.5 * images + 0.5
        images = TF.resize(images, [299, 299]).to(device)
        inception_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        inception_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        images = (images - inception_mean) / inception_std
        
        inception_score = metric.inception_score(torch.utils.data.DataLoader(images, batch_size=64))
        fid = metric.fid(torch.utils.data.DataLoader(images, batch_size=64))
        
        # (IS, IS_std), FID = get_inception_and_fid_score(
        #     images, FLAGS.fid_cache, num_images=FLAGS.num_images,
        #     use_torch=FLAGS.fid_use_torch, verbose=True)
        return inception_score, fid, images
    
    def train(self):
        dataset = CIFAR100("./data", train=True, download=True, 
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
        loader = DataLoader(dataset, batch_size=self.args.batch_size,
                            num_workers=self.args.num_workers, shuffle=True, pin_memory=True)
        datalooper = self.infiniteloop(loader)
        start_epoch = 1
        
        metric = Metrics(dataset)

        model = UNet().to(self.device)
        ema_model = copy.deepcopy(model).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        trainer = GaussianDiffusionTrainer(model, (self.args.beta_1, self.args.beta_T), self.args.T).to(self.device)
        sampler = DDIMSampler(model, (self.args.beta_1, self.args.beta_T), self.args.T).to(self.device)
        ema_sampler = DDIMSampler(ema_model, (self.args.beta_1, self.args.beta_T), self.args.T).to(self.device)
        
        os.makedirs(os.path.join(self.args.save_path, 'sample'))
        x_T = torch.randn(self.args.sample_size, 3, self.args.img_size, self.args.img_size)
        x_T = x_T.to(self.device)
        grid = (make_grid(next(iter(loader))[0][:self.args.sample_size]) + 1) / 2
        
        sample_step = self.args.sample_step
        save_step = self.args.save_step
        eval_step = self.args.eval_step
        
        start = time.time()
        step_start = time.time()

        for epoch in range(start_epoch, self.args.total_steps + 1):
            total_loss, total_num = 0., 0
            for images, _ in loader:
                optimizer.zero_grad()
                x_0 = images.to(self.device)

                loss = trainer(x_0)
                loss.backward()
                optimizer.step()
                
                self.ema(model, ema_model, self.args.ema_decay)
                
                total_loss += loss.item()
                total_num += x_0.shape[0]
            loss = total_loss / total_num
            
            self.logger.debug("%d/%d " % (epoch, self.args.total_steps) +
                f"loss: {loss:.4f}\ttime: {time_to_str(time.time() - step_start)}")
            step_start = time.time()
            
            # sample
            if sample_step > 0 and epoch % sample_step == 0:
                model.eval()
                with torch.no_grad():
                    x_0 = ema_sampler(x_T)
                    grid = (make_grid(x_0) + 1) / 2
                    path = os.path.join(self.args.save_path, 'sample', '%d.png' % epoch)
                    save_image(grid, path)
                    # writer.add_image('sample', grid, step)
                model.train()

            # save
            if save_step > 0 and epoch % save_step == 0:
                ckpt = {
                    'net_model': model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    # 'sched': sched.state_dict(),
                    'optim': optimizer.state_dict(),
                    'step': epoch,
                    'x_T': x_T,
                }
                torch.save(ckpt, os.path.join(self.args.save_path, 'ckpt.pt'))

            # evaluate
            if eval_step > 0 and epoch % eval_step == 0:
                net_IS, net_FID, _ = self.evaluate(self.args, sampler, model, self.device, metric)
                ema_IS, ema_FID, _ = self.evaluate(self.args, ema_sampler, ema_model, self.device, metric)
                metrics = {
                    'IS': net_IS.item(),
                    'FID': net_FID.item(),
                    'IS_EMA': ema_IS.item(),
                    'FID_EMA': ema_FID.item()
                }
                self.logger.debug(", ".join('%s:%.3f' % (k, v) for k, v in metrics.items()))
                # for name, value in metrics.items():
                #     writer.add_scalar(name, value, step)
                # writer.flush()
                with open(os.path.join(self.args.save_path, 'eval.txt'), 'a') as f:
                    metrics['step'] = epoch
                    f.write(json.dumps(metrics) + "\n")
        
        self.logger.debug(f"training time: {time_to_str(time.time() - start)}")

            # total_loss += loss.item()
            # total_num += x_0.shape[0]

            # data.set_description(f"Epoch: {epoch}")
            # data.set_postfix(ordered_dict={
            #     "train_loss": total_loss / total_num,
            # })
            # model_checkpoint.step(loss, model=model.state_dict(), config=config,
            #                     optimizer=optimizer.state_dict(), start_epoch=epoch,
            #                     model_checkpoint=model_checkpoint.state_dict())
    