import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import CIFAR100
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid, save_image

from model.DDIM.unet import UNet
from model.DDIM.diffusion import DDIMForwardTrainer, DDIMSampler, EMAHelper
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


class Distribution(torch.Tensor):
  # Init the params of the distribution
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
    # return self.variable
    
  # Silly hack: overwrite the to() method to wrap the new object
  # in a distribution as well
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
        
        
    def infiniteloop(self, dataloader):
        while True:
            for x, y in iter(dataloader):
                yield x, y
        
    
    def __load_data(self, batch_size):
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
        
        loader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=self.args.num_workers, shuffle=True, pin_memory=True, drop_last=True)
        return loader, inceptionset
    
    
    def __prepare_z_y(self, G_batch_size, dim_z, nclasses, device='cuda', 
                fp16=False,z_var=1.0):
        z_ = Distribution(torch.randn(G_batch_size, dim_z, requires_grad=False))
        z_.init_distribution('normal', mean=0, var=z_var)
        z_ = z_.to(device,torch.float16 if fp16 else torch.float32)   
        
        if fp16:
            z_ = z_.half()

        y_ = Distribution(torch.zeros(G_batch_size, requires_grad=False))
        y_.init_distribution('categorical',num_categories=nclasses)
        y_ = y_.to(device, torch.int64)
        return z_, y_
    
    
    # Sample function for use with inception metrics
    def sample(self, G, z_, y_):
        with torch.no_grad():
            z_.sample_()
            y_.sample_()
            G_z = G(z_, G.shared(y_))
            return G_z, y_
    
    
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
            'model': (model[0].state_dict(), model[1].state_dict()),
            'model_ema': model_ema,
            'optimizer': (optimizer[0].state_dict(), optimizer[1].state_dict()),
            'fixed_noise': fixed_noise
        }, os.path.join(save_path, model_name))
        
        
    def __load_model(self):
        self.logger.debug(f'Loading model from {os.getcwd() + self.args.load_model}')
        model_path = self.args.load_model
        data = torch.load(model_path)
        return data['epoch'], data['model'], data['model_ema'], data['optimizer'], data['fixed_noise']
        
    
    def __evaluate(self, G, metric, noise, steps, file_name=None):
        # self.ema_helper.ema(self.ema_model)
        G.eval()
        with torch.no_grad():
            generated_images = G(noise[0], G.shared(noise[1]))
            # print(generated_images.item(), type(generated_images.item()))
            generated_images = torch.FloatTensor(generated_images.tolist())
        G.train()
        
        if file_name is not None:
            # sample_image = torch.clamp(generated_images, -1.0, 1.0)
            grid = (make_grid(generated_images)[:64] + 1) / 2
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
    
    
    def loss_hinge_dis(self, dis_fake, dis_real):
        loss_real = torch.mean(F.relu(1. - dis_real))
        loss_fake = torch.mean(F.relu(1. + dis_fake))
        return loss_real, loss_fake
    
    
    def loss_hinge_gen(self, dis_fake):
        loss = -torch.mean(dis_fake)
        return loss
    
    
    # Apply modified ortho reg to a model
    # This function is an optimized version that directly computes the gradient,
    # instead of computing and then differentiating the loss.
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
        
        
    def GAN_training_function(self, G, D, GD, z_, y_):
        def toggle_grad(model, on_or_off):
            for param in model.parameters():
                param.requires_grad = on_or_off
        
        def train(x, y):
            G.optim.zero_grad()
            D.optim.zero_grad()
            # How many chunks to split x and y into?
            x = torch.split(x, self.args.batch_size)
            y = torch.split(y, self.args.batch_size)
            counter = 0
            
            # Optionally toggle D and G's "require_grad"
            # if config['toggle_grads']:
            #     toggle_grad(D, True)
            #     toggle_grad(G, False)
            toggle_grad(D, True)
            toggle_grad(G, False)
            
            for step_index in range(self.args.num_D_steps):
            # for step_index in range(2):
                # If accumulating gradients, loop multiple times before an optimizer step
                D.optim.zero_grad()
                for accumulation_index in range(self.args.num_D_accumulations):
                # for accumulation_index in range(1):
                    z_.sample_()
                    y_.sample_()
                    # print('dfdffd',counter)
                    D_fake, D_real = GD(z_[:self.args.batch_size], y_[:self.args.batch_size], 
                                        x[counter], y[counter], train_G=False, 
                                        split_D=True)
                    
                    # Compute components of D's loss, average them, and divide by 
                    # the number of gradient accumulations
                    D_loss_real, D_loss_fake = self.loss_hinge_dis(D_fake, D_real)
                    # D_loss = (D_loss_real + D_loss_fake) / float(config['num_D_accumulations'])
                    D_loss = (D_loss_real + D_loss_fake) / float(1)
                    D_loss.backward()
                    counter += 1
                
            # Optionally apply ortho reg in D
            # if config['D_ortho'] > 0.0:
            #     # Debug print to indicate we're using ortho reg in D.
            #     print('using modified ortho reg in D')
            #     self.ortho(D, config['D_ortho'])
            
            D.optim.step()
            
            # Optionally toggle "requires_grad"
            # if config['toggle_grads']:
            #     toggle_grad(D, False)
            #     toggle_grad(G, True)
            toggle_grad(D, False)
            toggle_grad(G, True)
            
            # Zero G's gradients by default before training G, for safety
            G.optim.zero_grad()
            
            # If accumulating gradients, loop multiple times
            # for accumulation_index in range(config['num_G_accumulations']):    
            for accumulation_index in range(1):    
                z_.sample_()
                y_.sample_()
                # D_fake = GD(z_, y_, train_G=True, split_D=config['split_D'])
                D_fake = GD(z_, y_, train_G=True, split_D=False)
                G_loss = self.loss_hinge_gen(D_fake) / float(1)
                G_loss.backward()
            
            # Optionally apply modified ortho reg in G
            # if config['G_ortho'] > 0.0:
            #     print('using modified ortho reg in G') # Debug print to indicate we're using ortho reg in G
            #     # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
            #     self.ortho(G, config['G_ortho'], 
            #                 blacklist=[param for param in G.shared.parameters()])
            G.optim.step()
            
            # If we have an ema, update it, regardless of if we test with it or not
            # if config['ema']:
            #     ema.update(state_dict['itr'])
            self.ema.update(GD.G)
            
            out = {'G_loss': float(G_loss.item()), 
                    'D_loss_real': float(D_loss_real.item()),
                    'D_loss_fake': float(D_loss_fake.item())}
            # Return G's loss and the components of D's loss.
            return out
        return train
        
        
    def train(self):
        # Update the config dict as necessary
        # This is for convenience, to add settings derived from the user-specified
        # configuration into the config-dict (e.g. inferring the number of classes
        # and size of the images from the dataset, passing in a pytorch object
        # for the activation specified as a string)
        self.args.resolution = 32
        self.args.n_classes = 100
        self.args.G_activation = nn.ReLU(inplace=False)
        self.args.D_activation = nn.ReLU(inplace=False)
        # By default, skip init if resuming training.
        # if config['resume']:
        #     print('Skipping initialization for training resumption...')
        #     config['skip_init'] = True
        # config = utils.update_config_roots(config)
        device = self.device
        
        # Seed RNG
        # utils.seed_rng(config['seed'])

        # Prepare root folders if necessary
        # utils.prepare_root(config)

        # Setup cudnn.benchmark for free speed
        # torch.backends.cudnn.benchmark = True

        # Import the model--this line allows us to dynamically select different files.
        # model = __import__(config['model'])
        # experiment_name = (config['experiment_name'] if config['experiment_name']
        #                     else utils.name_from_config(config))
        # print('Experiment name is %s' % experiment_name)

        # Next, build the model
        G = Generator(**vars(self.args)).to(device)
        D = Discriminator(**vars(self.args)).to(device)
        
        # If using EMA, prepare it
        # if config['ema']:
        #     print('Preparing EMA for G with decay of {}'.format(config['ema_decay']))
        #     G_ema = model.Generator(**{**config, 'skip_init':True, 
        #                             'no_optim': True}).to(device)
        #     ema = utils.ema(G, G_ema, config['ema_decay'], config['ema_start'])
        # else:
        #     G_ema, ema = None, None
        print('Preparing EMA for G with decay of {}'.format(self.args.ema_decay))
        G_ema = Generator(**{**vars(self.args), 'skip_init':True, 
                                'no_optim': True}).to(device)
        self.ema = EMAHelper(mu=self.args.ema_decay, device=self.device)
        self.ema.register(G_ema)
        # ema = utils.ema(G, G_ema, config['ema_decay'], config['ema_start'])
        
        # FP16?
        # if config['G_fp16']:
        #     print('Casting G to float16...')
        #     G = G.half()
        #     if config['ema']:
        #     G_ema = G_ema.half()
        # if config['D_fp16']:
        #     print('Casting D to fp16...')
        #     D = D.half()
            # Consider automatically reducing SN_eps?
        GD = G_D(G, D)
        print(G)
        print(D)
        print('Number of params in G: {} D: {}'.format(
            *[sum([p.data.nelement() for p in net.parameters()]) for net in [G,D]]))
        # Prepare state dict, which holds things like epoch # and itr #
        state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                        'best_IS': 0, 'best_FID': 999999, 'config': vars(self.args)}

        # If loading from a pre-trained model, load weights
        # if config['resume']:
        #     print('Loading weights...')
        #     utils.load_weights(G, D, state_dict,
        #                     config['weights_root'], experiment_name, 
        #                     config['load_weights'] if config['load_weights'] else None,
        #                     G_ema if config['ema'] else None)

        # If parallel, parallelize the GD module
        # if config['parallel']:
        #     GD = nn.DataParallel(GD)
        #     if config['cross_replica']:
        #     patch_replication_callback(GD)

        # Prepare loggers for stats; metrics holds test metrics,
        # lmetrics holds any desired training metrics.
        self.args.logs_root = self.args.save_path
        # test_metrics_fname = '%s/metric.jsonl' % (self.args.logs_root)
        # train_metrics_fname = '%s/metric' % (config['logs_root'])
        # print('Inception Metrics will be saved to {}'.format(test_metrics_fname))
        # test_log = utils.MetricsLogger(test_metrics_fname, 
        #                                 reinitialize=(not config['resume']))
        # print('Training Metrics will be saved to {}'.format(train_metrics_fname))
        # train_log = utils.MyLogger(train_metrics_fname, 
        #                             reinitialize=(not config['resume']),
        #                             logstyle=config['logstyle'])
        # Write metadata
        # utils.write_metadata(config['logs_root'], experiment_name, config, state_dict)
        # Prepare data; the Discriminator's batch size is all that needs to be passed
        # to the dataloader, as G doesn't require dataloading.
        # Note that at every loader iteration we pass in enough data to complete
        # a full D iteration (regardless of number of D steps and accumulations)
        # D_batch_size = (self.args.batch_size * self.args.num_D_steps * self.args.num_D_steps.num_D_accumulations)
        # loaders = utils.get_data_loaders(**{**vars(self.args), 'batch_size': D_batch_size,
        #                                     'start_itr': state_dict['itr']})
        
        D_batch_size = (self.args.batch_size * self.args.num_D_steps * self.args.num_D_accumulations)
        loaders = self.__load_data(D_batch_size)

        # Prepare inception metrics: FID and IS
        # get_inception_metrics = inception_utils.prepare_inception_metrics(config['dataset'], config['parallel'], config['no_fid'])
        metric = Metrics(loaders[1])

        # Prepare noise and randomly sampled label arrays
        # Allow for different batch sizes in G
        G_batch_size = self.args.batch_size
        z_, y_ = self.__prepare_z_y(G_batch_size, G.dim_z, self.args.n_classes, device=device)
        # Prepare a fixed z & y to see individual sample evolution throghout training
        fixed_z, fixed_y = self.__prepare_z_y(G_batch_size, G.dim_z, self.args.n_classes, device=device)  
        fixed_z.sample_()
        fixed_y.sample_()
        # Loaders are loaded, prepare the training 
        # function
        # if config['which_train_fn'] == 'GAN':
        #     train = train_fns.GAN_training_function(G, D, GD, z_, y_, 
        #                                             ema, state_dict, config)
        # # Else, assume debugging and use the dummy train fn
        # else:
        #     train = train_fns.dummy_training_function()
        train = self.GAN_training_function(G, D, GD, z_, y_)
        # Prepare Sample function for use with inception metrics
        sample = functools.partial(self.sample,
                                    G=self.ema.ema_copy(G_ema),
                                    z_=z_, y_=y_)

        print('Beginning training at epoch %d...' % state_dict['epoch'])
        # Train for specified number of epochs, although we mostly track G iterations.
        # for epoch in range(state_dict['epoch'], config['num_epochs']):    
        
        highest_is = 0
        lowest_fid = 1e10
        for epoch in range(0, self.args.num_epochs):    
            # Which progressbar to use? TQDM or my own?
            # if self.args.pbar == 'mine':
            #     pbar = utils.progress(loaders[0],displaytype='s1k' if self.args.use_multiepoch_sampler else 'eta')
            # else:
            #     pbar = tqdm(loaders[0])
            # for i, (x, y) in enumerate(pbar):
            for i, (x, y) in enumerate(tqdm(loaders[0], ncols=0)):
                # Increment the iteration counter
                state_dict['itr'] += 1
                # Make sure G and D are in training mode, just in case they got set to eval
                # For D, which typically doesn't have BN, this shouldn't matter much.
                G.train()
                D.train()
                # if config['ema']:
                #     G_ema.train()
                # G_ema.train()
                # if config['D_fp16']:
                #     x, y = x.to(device).half(), y.to(device)
                # else:
                #     x, y = x.to(device), y.to(device)
                x, y = x.to(device), y.to(device)
                metrics = train(x, y)
                
                self.logger.debug(f"itr: {state_dict['itr']}, G_loss: {metrics['G_loss']:.4f}, D_loss_real: {metrics['D_loss_real']:.4f}, D_loss_fake: {metrics['D_loss_fake']:.4f}")
                
                if state_dict['itr'] % 500 or state_dict['itr'] == self.args.total_steps - 1:
                    inception_score, fid = self.__evaluate(GD.G, metric, [fixed_z, fixed_y], file_name=f'{state_dict['itr']}.png',
                                                        steps=state_dict['itr'])
                    
                    model_name = f'{state_dict['itr']}_{inception_score:.04f}_{fid:.04f}.pt'
                    if inception_score > highest_is:
                        highest_is = inception_score
                        self.__save_model('is', state_dict['itr'], model_name, [G, D], self.ema.state_dict(), [G.optim, D.optim], [fixed_z, fixed_y])
                        self.logger.debug(f'highest IS')
                    if fid < lowest_fid:
                        lowest_fid = fid
                        self.__save_model('fid', state_dict['itr'], model_name, [G, D], self.ema.state_dict(), [G.optim, D.optim], [fixed_z, fixed_y])
                        self.logger.debug(f'lowest FID')
                    
                self.__save_model('model', state_dict['itr'], 'last.pt', [G, D], self.ema.state_dict(), [G.optim, D.optim], [fixed_z, fixed_y])
                
                # train_log.log(itr=int(state_dict['itr']), **metrics)
                
                # Every sv_log_interval, log singular values
                # if (config['sv_log_interval'] > 0) and (not (state_dict['itr'] % config['sv_log_interval'])):
                #     train_log.log(itr=int(state_dict['itr']), 
                #                 **{**utils.get_SVs(G, 'G'), **utils.get_SVs(D, 'D')})

                # If using my progbar, print metrics.
                # if config['pbar'] == 'mine':
                #     print(', '.join(['itr: %d' % state_dict['itr']] 
                #                     + ['%s : %+4.3f' % (key, metrics[key])
                #                     for key in metrics]), end=' ')

                # Save weights and copies as configured at specified interval
                # if not (state_dict['itr'] % config['save_every']):
                #     if config['G_eval_mode']:
                #     print('Switchin G to eval mode...')
                #     G.eval()
                #     if config['ema']:
                #         G_ema.eval()
                #     train_fns.save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y, 
                #                             state_dict, config, experiment_name)

                # Test every specified interval
                # if not (state_dict['itr'] % config['test_every']):
                #     if config['G_eval_mode']:
                #     print('Switchin G to eval mode...')
                #     G.eval()
                #     train_fns.test(G, D, G_ema, z_, y_, state_dict, config, sample,
                #                 get_inception_metrics, experiment_name, test_log)
            # Increment epoch counter at end of epoch
            # state_dict['epoch'] += 1


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
            _, model_state, model_ema_state, optimizer_state, fixed_noise =  self.__load_model()
            self.unet_model.load_state_dict(model_state)
            self.ema_model.load_state_dict(model_ema_state)
            optimizer.load_state_dict(optimizer_state)
            
            self.ema_helper.register(self.ema_model)
        
        self.sampler = DDIMSampler(self.ema_model, self.args.beta_1, self.args.beta_T, self.args.T).to(self.device)
        
        fixed_noise_loader = DataLoader(fixed_noise, batch_size=self.args.batch_size * 2,
                            num_workers=0, pin_memory=True)
        
        self.__evaluate(metric, fixed_noise_loader, file_name=f'sample.png',
                            steps=self.args.sample_step, method=self.args.method,
                            eta=self.args.eta, only_return_x_0=True)
