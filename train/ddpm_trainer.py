import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import copy
import json
import os

from torchvision.datasets import CIFAR100
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import trange

from model.DDPM import UNet

import argparse

from utils import get_default_device
from metrics import Metrics
from logger.main_logger import MainLogger


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, img_size=32,
                 mean_type='eps', var_type='fixedlarge'):
        assert mean_type in ['xprev' 'xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model = model
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)

        # Mean parameterization
        if self.mean_type == 'xprev':       # the model predicts x_{t-1}
            x_prev = self.model(x_t, t)
            x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
            model_mean = x_prev
        elif self.mean_type == 'xstart':    # the model predicts x_0
            x_0 = self.model(x_t, t)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        elif self.mean_type == 'epsilon':   # the model predicts epsilon
            eps = self.model(x_t, t)
            x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        else:
            raise NotImplementedError(self.mean_type)
        x_0 = torch.clip(x_0, -1., 1.)

        return model_mean, model_log_var

    def forward(self, x_T):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, log_var = self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.exp(0.5 * log_var) * noise
        x_0 = x_t
        return torch.clip(x_0, -1, 1)



def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x


def warmup_lr(step):
    return min(step, 5000) / 5000


def evaluate(args: argparse.Namespace, sampler, model, device, metric: Metrics):
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


def train(args: argparse.Namespace):
    logger = MainLogger(args)
    
    device = get_default_device()
    
    # dataset
    dataset = CIFAR100("./data", train=True, download=True, 
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True)
    datalooper = infiniteloop(dataloader)
    
    metric = Metrics(dataset)

    # model setup
    net_model = UNet(
        T=args.T, ch=args.ch, ch_mult=args.ch_mult, attn=args.attn,
        num_res_blocks=args.num_res_blocks, dropout=args.dropout).to(device)
    ema_model = copy.deepcopy(net_model).to(device)
    optim = torch.optim.Adam(net_model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    trainer = GaussianDiffusionTrainer(
        net_model, args.beta_1, args.beta_T, args.T).to(device)
    net_sampler = GaussianDiffusionSampler(
        net_model, args.beta_1, args.beta_T, args.T, args.img_size,
        args.mean_type, args.var_type).to(device)
    ema_sampler = GaussianDiffusionSampler(
        ema_model, args.beta_1, args.beta_T, args.T, args.img_size,
        args.mean_type, args.var_type).to(device)
    # if args.parallel:
    #     trainer = torch.nn.DataParallel(trainer)
    #     net_sampler = torch.nn.DataParallel(net_sampler)
    #     ema_sampler = torch.nn.DataParallel(ema_sampler)
    
    # log setup
    os.makedirs(os.path.join(args.save_path, 'sample'))
    x_T = torch.randn(args.sample_size, 3, args.img_size, args.img_size)
    x_T = x_T.to(device)
    grid = (make_grid(next(iter(dataloader))[0][:args.sample_size]) + 1) / 2
    # writer = SummaryWriter(args.logdir)
    # writer.add_image('real_sample', grid)
    # writer.flush()
    # backup all arguments
    with open(os.path.join(args.save_path, "flagfile.txt"), 'w') as f:
        f.write(str(vars(args)))
    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))

    # start training
    with trange(args.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            # train
            optim.zero_grad()
            x_0 = next(datalooper).to(device)
            loss = trainer(x_0).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), args.grad_clip)
            optim.step()
            sched.step()
            ema(net_model, ema_model, args.ema_decay)

            # log
            # writer.add_scalar('loss', loss, step)
            pbar.set_postfix(loss='%.3f' % loss)

            # sample
            if args.sample_step > 0 and step % args.sample_step == 0:
                net_model.eval()
                with torch.no_grad():
                    x_0 = ema_sampler(x_T)
                    grid = (make_grid(x_0) + 1) / 2
                    path = os.path.join(args.save_path, 'sample', '%d.png' % step)
                    save_image(grid, path)
                    # writer.add_image('sample', grid, step)
                net_model.train()

            # save
            if args.save_step > 0 and step % args.save_step == 0:
                ckpt = {
                    'net_model': net_model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'sched': sched.state_dict(),
                    'optim': optim.state_dict(),
                    'step': step,
                    'x_T': x_T,
                }
                torch.save(ckpt, os.path.join(args.save_path, 'ckpt.pt'))

            # evaluate
            if args.eval_step > 0 and step % args.eval_step == 0:
                net_IS, net_FID, _ = evaluate(args, net_sampler, net_model, device, metric)
                ema_IS, ema_FID, _ = evaluate(args, ema_sampler, ema_model, device, metric)
                metrics = {
                    'IS': net_IS.item(),
                    'FID': net_FID.item(),
                    'IS_EMA': ema_IS.item(),
                    'FID_EMA': ema_FID.item()
                }
                logger.debug(
                    "%d/%d " % (step, args.total_steps) +
                    ", ".join('%s:%.3f' % (k, v) for k, v in metrics.items()))
                # for name, value in metrics.items():
                #     writer.add_scalar(name, value, step)
                # writer.flush()
                with open(os.path.join(args.save_path, 'eval.txt'), 'a') as f:
                    metrics['step'] = step
                    f.write(json.dumps(metrics) + "\n")
    # writer.close()
