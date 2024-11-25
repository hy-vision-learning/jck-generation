import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

import preprocess.cgan_data_preprocessor
import preprocess.ddim_preprocessor
import train
from model import DCGAN, CGAN
import preprocess

import numpy as np

import argparse
import logging
from datetime import datetime

import os
import sys
import random

from change_randomseed import RANDOMSEED

from logger.main_logger import MainLogger
from enums import ModelEnum
import train.biggan_trainer
import train.ddim_trainer
import train.ddpm_trainer

torch.autograd.set_detect_anomaly(True)


random.seed(RANDOMSEED)
os.environ["PYTHONHASHSEED"] = str(RANDOMSEED)
np.random.seed(RANDOMSEED)
torch.manual_seed(RANDOMSEED)
torch.cuda.manual_seed_all(RANDOMSEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_arg_parse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-t', '--test', type=int, help='테스트모드', default=0)
    parser.add_argument('-pm', '--model_path', type=str, help='모델 폴더 이름', default='')
    
    parser.add_argument('-lf', '--log_file', type=int, help='로그 파일 출력 여부. 0=false, 1=true', default=1)
    
    parser.add_argument('-m', '--model', type=ModelEnum, help='학습 모델', choices=list(ModelEnum), default=ModelEnum.DCGAN)
    # parser.add_argument('--load_model', type=str, help='로드할 모델', default=None)
    
    # parser.add_argument('-w', '--num_worker', type=int, help='DataLoader worker', default=0)
    # parser.add_argument('-b', '--batch_size', type=int, help='학습 배치사이즈', default=128)
    
    # parser.add_argument('-e', '--epoch', type=int, help='epoch', default=100)
    # parser.add_argument('-mlr', '--max_learning_rate', type=float, help='optimizer max learning rate 설정', default=0.1)
    # parser.add_argument('-milr', '--min_learning_rate', type=float, help='optimizer min learning rate 설정', default=1e-4)
    # parser.add_argument('-wd', '--weight_decay', type=float, help='optimizer weight decay 설정', default=0.0)
    # parser.add_argument('-snt', '--nesterov', type=int, help="nesterov sgd 사용 여부", default=1)
    
    # Unet
    # parser.add_argument('--ch', type=int, help='base channel of UNet', default=128)
    # parser.add_argument('--ch_mult', nargs='+', type=int, help='channel multiplier', default=[1, 2, 2, 2])
    # parser.add_argument('--attn', nargs='+', type=int, help='add attention to these levels', default=[1])
    # parser.add_argument('--num_res_blocks', type=int, help='# resblock in each level', default=2)
    # parser.add_argument('--dropout', type=float, help='dropout rate of resblock', default=0.1)

    # Gaussian Diffusion
    # parser.add_argument('--beta_1', type=float, help='start beta value', default=1e-4)
    # parser.add_argument('--beta_T', type=float, help='end beta value', default=0.02)
    # parser.add_argument('--T', type=int, help='total diffusion steps', default=1000)
    # parser.add_argument('--method', type=str, choices=['linear', 'cosine', 'quadratic'], help='beta time scheduler', default='linear')
    # parser.add_argument('--mean_type', type=str, choices=['xprev', 'xstart', 'epsilon'], help='predict variable', default='epsilon')
    # parser.add_argument('--var_type', type=str, choices=['fixedlarge', 'fixedsmall'], help='variance type', default='fixedlarge')

    # Training
    # parser.add_argument('--lr', type=float, help='target learning rate', default=2e-4)
    # parser.add_argument('--grad_clip', type=float, help="gradient norm clipping", default=-1.0)
    # parser.add_argument('--total_steps', type=int, help='total training steps', default=800000)
    # parser.add_argument('--img_size', type=int, help='image size', default=32)
    # parser.add_argument('--warmup', type=int, help='learning rate warmup', default=5000)
    # parser.add_argument('--batch_size', type=int, help='batch size', default=128)
    # parser.add_argument('--num_workers', type=int, help='workers of Dataloader', default=4)
    # parser.add_argument('--ema_decay', type=float, help="ema decay rate", default=0.9999)
    # parser.add_argument('--parallel', type=bool, help='multi gpu training', default=False)

    # Logging & Sampling
    # parser.add_argument('--logdir', type=str, help='log directory', default='./logs/DDPM_CIFAR10_EPS')
    # parser.add_argument('--sample_size', type=int, help="sampling size of images", default=64)
    # parser.add_argument('--sample_step', type=int, help='frequency of sampling', default=1000)
    # parser.add_argument('--eta', type=float, help='sampler eta', default=0.0)
    # parser.add_argument('--load_model', type=str, help='load pretrained model path', default=None)

    # Evaluation
    # parser.add_argument('--save_step', type=int, help='frequency of saving checkpoints, 0 to disable during training', default=5000)
    # parser.add_argument('--eval_step', type=int, help='frequency of evaluating model, 0 to disable during training', default=0)
    # parser.add_argument('--num_images', type=int, help='the number of generated images for evaluation', default=50000)
    # parser.add_argument('--eval_sample_step', type=int, help='frequency of sampling', default=500)
    # parser.add_argument('--fid_use_torch', type=bool, help='calculate IS and FID on gpu', default=False)
    # parser.add_argument('--fid_cache', type=str, help='FID cache', default='./stats/cifar10.train.npz')
    
    
    parser.add_argument(
    '--dataset', type=str, default='I128_hdf5',
    help='Which Dataset to train on, out of I128, I256, C10, C100;'
         'Append "_hdf5" to use the hdf5 version for ISLVRC '
         '(default: %(default)s)')
    parser.add_argument(
        '--augment', action='store_true', default=False,
        help='Augment with random crops and flips (default: %(default)s)')
    parser.add_argument(
        '--num_workers', type=int, default=8,
        help='Number of dataloader workers; consider using less for HDF5 '
            '(default: %(default)s)')
    parser.add_argument(
        '--no_pin_memory', action='store_false', dest='pin_memory', default=True,
        help='Pin data into memory through dataloader? (default: %(default)s)') 
    parser.add_argument(
        '--shuffle', action='store_true', default=False,
        help='Shuffle the data (strongly recommended)? (default: %(default)s)')
    parser.add_argument(
        '--load_in_mem', action='store_true', default=False,
        help='Load all data into memory? (default: %(default)s)')
    parser.add_argument(
        '--use_multiepoch_sampler', action='store_true', default=False,
        help='Use the multi-epoch sampler for dataloader? (default: %(default)s)')
    
    
    ### Model stuff ###
    # parser.add_argument(
    #     '--model', type=str, default='BigGAN',
    #     help='Name of the model module (default: %(default)s)')
    parser.add_argument(
        '--G_param', type=str, default='SN',
        help='Parameterization style to use for G, spectral norm (SN) or SVD (SVD)'
            ' or None (default: %(default)s)')
    parser.add_argument(
        '--D_param', type=str, default='SN',
        help='Parameterization style to use for D, spectral norm (SN) or SVD (SVD)'
            ' or None (default: %(default)s)')    
    parser.add_argument(
        '--G_ch', type=int, default=64,
        help='Channel multiplier for G (default: %(default)s)')
    parser.add_argument(
        '--D_ch', type=int, default=64,
        help='Channel multiplier for D (default: %(default)s)')
    parser.add_argument(
        '--G_depth', type=int, default=1,
        help='Number of resblocks per stage in G? (default: %(default)s)')
    parser.add_argument(
        '--D_depth', type=int, default=1,
        help='Number of resblocks per stage in D? (default: %(default)s)')
    parser.add_argument(
        '--D_thin', action='store_false', dest='D_wide', default=True,
        help='Use the SN-GAN channel pattern for D? (default: %(default)s)')
    parser.add_argument(
        '--G_shared', action='store_true', default=False,
        help='Use shared embeddings in G? (default: %(default)s)')
    parser.add_argument(
        '--shared_dim', type=int, default=0,
        help='G''s shared embedding dimensionality; if 0, will be equal to dim_z. '
            '(default: %(default)s)')
    parser.add_argument(
        '--dim_z', type=int, default=128,
        help='Noise dimensionality: %(default)s)')
    parser.add_argument(
        '--z_var', type=float, default=1.0,
        help='Noise variance: %(default)s)')    
    parser.add_argument(
        '--hier', action='store_true', default=False,
        help='Use hierarchical z in G? (default: %(default)s)')
    parser.add_argument(
        '--cross_replica', action='store_true', default=False,
        help='Cross_replica batchnorm in G?(default: %(default)s)')
    parser.add_argument(
        '--mybn', action='store_true', default=False,
        help='Use my batchnorm (which supports standing stats?) %(default)s)')
    parser.add_argument(
        '--G_nl', type=str, default='relu',
        help='Activation function for G (default: %(default)s)')
    parser.add_argument(
        '--D_nl', type=str, default='relu',
        help='Activation function for D (default: %(default)s)')
    parser.add_argument(
        '--G_attn', type=str, default='64',
        help='What resolutions to use attention on for G (underscore separated) '
            '(default: %(default)s)')
    parser.add_argument(
        '--D_attn', type=str, default='64',
        help='What resolutions to use attention on for D (underscore separated) '
            '(default: %(default)s)')
    parser.add_argument(
        '--norm_style', type=str, default='bn',
        help='Normalizer style for G, one of bn [batchnorm], in [instancenorm], '
            'ln [layernorm], gn [groupnorm] (default: %(default)s)')
            
    ### Model init stuff ###
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed to use; affects both initialization and '
            ' dataloading. (default: %(default)s)')
    parser.add_argument(
        '--G_init', type=str, default='ortho',
        help='Init style to use for G (default: %(default)s)')
    parser.add_argument(
        '--D_init', type=str, default='ortho',
        help='Init style to use for D(default: %(default)s)')
    parser.add_argument(
        '--skip_init', action='store_true', default=False,
        help='Skip initialization, ideal for testing when ortho init was used '
            '(default: %(default)s)')
    
    ### Optimizer stuff ###
    parser.add_argument(
        '--G_lr', type=float, default=5e-5,
        help='Learning rate to use for Generator (default: %(default)s)')
    parser.add_argument(
        '--D_lr', type=float, default=2e-4,
        help='Learning rate to use for Discriminator (default: %(default)s)')
    parser.add_argument(
        '--G_B1', type=float, default=0.0,
        help='Beta1 to use for Generator (default: %(default)s)')
    parser.add_argument(
        '--D_B1', type=float, default=0.0,
        help='Beta1 to use for Discriminator (default: %(default)s)')
    parser.add_argument(
        '--G_B2', type=float, default=0.999,
        help='Beta2 to use for Generator (default: %(default)s)')
    parser.add_argument(
        '--D_B2', type=float, default=0.999,
        help='Beta2 to use for Discriminator (default: %(default)s)')
        
    ### Batch size, parallel, and precision stuff ###
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='Default overall batchsize (default: %(default)s)')
    parser.add_argument(
        '--G_batch_size', type=int, default=0,
        help='Batch size to use for G; if 0, same as D (default: %(default)s)')
    parser.add_argument(
        '--num_G_accumulations', type=int, default=1,
        help='Number of passes to accumulate G''s gradients over '
            '(default: %(default)s)')  
    parser.add_argument(
        '--num_D_steps', type=int, default=2,
        help='Number of D steps per G step (default: %(default)s)')
    parser.add_argument(
        '--num_D_accumulations', type=int, default=1,
        help='Number of passes to accumulate D''s gradients over '
            '(default: %(default)s)')
    parser.add_argument(
        '--split_D', action='store_true', default=False,
        help='Run D twice rather than concatenating inputs? (default: %(default)s)')
    parser.add_argument(
        '--num_epochs', type=int, default=100,
        help='Number of epochs to train for (default: %(default)s)')
    parser.add_argument(
        '--parallel', action='store_true', default=False,
        help='Train with multiple GPUs (default: %(default)s)')
    parser.add_argument(
        '--G_fp16', action='store_true', default=False,
        help='Train with half-precision in G? (default: %(default)s)')
    parser.add_argument(
        '--D_fp16', action='store_true', default=False,
        help='Train with half-precision in D? (default: %(default)s)')
    parser.add_argument(
        '--D_mixed_precision', action='store_true', default=False,
        help='Train with half-precision activations but fp32 params in D? '
            '(default: %(default)s)')
    parser.add_argument(
        '--G_mixed_precision', action='store_true', default=False,
        help='Train with half-precision activations but fp32 params in G? '
            '(default: %(default)s)')
    parser.add_argument(
        '--accumulate_stats', action='store_true', default=False,
        help='Accumulate "standing" batchnorm stats? (default: %(default)s)')
    parser.add_argument(
        '--num_standing_accumulations', type=int, default=16,
        help='Number of forward passes to use in accumulating standing stats? '
            '(default: %(default)s)')        
        
    ### Bookkeping stuff ###  
    parser.add_argument(
        '--G_eval_mode', action='store_true', default=False,
        help='Run G in eval mode (running/standing stats?) at sample/test time? '
            '(default: %(default)s)')
    parser.add_argument(
        '--save_every', type=int, default=2000,
        help='Save every X iterations (default: %(default)s)')
    parser.add_argument(
        '--num_save_copies', type=int, default=2,
        help='How many copies to save (default: %(default)s)')
    parser.add_argument(
        '--num_best_copies', type=int, default=2,
        help='How many previous best checkpoints to save (default: %(default)s)')
    parser.add_argument(
        '--which_best', type=str, default='IS',
        help='Which metric to use to determine when to save new "best"'
            'checkpoints, one of IS or FID (default: %(default)s)')
    parser.add_argument(
        '--no_fid', action='store_true', default=False,
        help='Calculate IS only, not FID? (default: %(default)s)')
    parser.add_argument(
        '--test_every', type=int, default=5000,
        help='Test every X iterations (default: %(default)s)')
    parser.add_argument(
        '--num_inception_images', type=int, default=50000,
        help='Number of samples to compute inception metrics with '
            '(default: %(default)s)')
    parser.add_argument(
        '--hashname', action='store_true', default=False,
        help='Use a hash of the experiment name instead of the full config '
            '(default: %(default)s)') 
    parser.add_argument(
        '--base_root', type=str, default='',
        help='Default location to store all weights, samples, data, and logs '
            ' (default: %(default)s)')
    parser.add_argument(
        '--data_root', type=str, default='data',
        help='Default location where data is stored (default: %(default)s)')
    parser.add_argument(
        '--weights_root', type=str, default='weights',
        help='Default location to store weights (default: %(default)s)')
    parser.add_argument(
        '--logs_root', type=str, default='logs',
        help='Default location to store logs (default: %(default)s)')
    parser.add_argument(
        '--samples_root', type=str, default='samples',
        help='Default location to store samples (default: %(default)s)')  
    parser.add_argument(
        '--pbar', type=str, default='mine',
        help='Type of progressbar to use; one of "mine" or "tqdm" '
            '(default: %(default)s)')
    parser.add_argument(
        '--name_suffix', type=str, default='',
        help='Suffix for experiment name for loading weights for sampling '
            '(consider "best0") (default: %(default)s)')
    parser.add_argument(
        '--experiment_name', type=str, default='',
        help='Optionally override the automatic experiment naming with this arg. '
            '(default: %(default)s)')
    parser.add_argument(
        '--config_from_name', action='store_true', default=False,
        help='Use a hash of the experiment name instead of the full config '
            '(default: %(default)s)')
            
    ### EMA Stuff ###
    parser.add_argument(
        '--ema', action='store_true', default=False,
        help='Keep an ema of G''s weights? (default: %(default)s)')
    parser.add_argument(
        '--ema_decay', type=float, default=0.9999,
        help='EMA decay rate (default: %(default)s)')
    parser.add_argument(
        '--use_ema', action='store_true', default=False,
        help='Use the EMA parameters of G for evaluation? (default: %(default)s)')
    parser.add_argument(
        '--ema_start', type=int, default=0,
        help='When to start updating the EMA weights (default: %(default)s)')
    
    ### Numerical precision and SV stuff ### 
    parser.add_argument(
        '--adam_eps', type=float, default=1e-8,
        help='epsilon value to use for Adam (default: %(default)s)')
    parser.add_argument(
        '--BN_eps', type=float, default=1e-5,
        help='epsilon value to use for BatchNorm (default: %(default)s)')
    parser.add_argument(
        '--SN_eps', type=float, default=1e-8,
        help='epsilon value to use for Spectral Norm(default: %(default)s)')
    parser.add_argument(
        '--num_G_SVs', type=int, default=1,
        help='Number of SVs to track in G (default: %(default)s)')
    parser.add_argument(
        '--num_D_SVs', type=int, default=1,
        help='Number of SVs to track in D (default: %(default)s)')
    parser.add_argument(
        '--num_G_SV_itrs', type=int, default=1,
        help='Number of SV itrs in G (default: %(default)s)')
    parser.add_argument(
        '--num_D_SV_itrs', type=int, default=1,
        help='Number of SV itrs in D (default: %(default)s)')
    
    ### Ortho reg stuff ### 
    parser.add_argument(
        '--G_ortho', type=float, default=0.0, # 1e-4 is default for BigGAN
        help='Modified ortho reg coefficient in G(default: %(default)s)')
    parser.add_argument(
        '--D_ortho', type=float, default=0.0,
        help='Modified ortho reg coefficient in D (default: %(default)s)')
    parser.add_argument(
        '--toggle_grads', action='store_true', default=True,
        help='Toggle D and G''s "requires_grad" settings when not training them? '
            ' (default: %(default)s)')
    
    ### Which train function ###
    parser.add_argument(
        '--which_train_fn', type=str, default='GAN',
        help='How2trainyourbois (default: %(default)s)')  
    
    ### Resume training stuff
    parser.add_argument(
        '--load_weights', type=str, default='',
        help='Suffix for which weights to load (e.g. best0, copy0) '
            '(default: %(default)s)')
    parser.add_argument(
        '--resume', action='store_true', default=False,
        help='Resume training? (default: %(default)s)')
    
    ### Log stuff ###
    parser.add_argument(
        '--logstyle', type=str, default='%3.3e',
        help='What style to use when logging training metrics?'
            'One of: %#.#f/ %#.#e (float/exp, text),'
            'pickle (python pickle),'
            'npz (numpy zip),'
            'mat (MATLAB .mat file) (default: %(default)s)')
    parser.add_argument(
        '--log_G_spectra', action='store_true', default=False,
        help='Log the top 3 singular values in each SN layer in G? '
            '(default: %(default)s)')
    parser.add_argument(
        '--log_D_spectra', action='store_true', default=False,
        help='Log the top 3 singular values in each SN layer in D? '
            '(default: %(default)s)')
    parser.add_argument(
        '--sv_log_interval', type=int, default=10,
        help='Iteration interval for logging singular values '
            ' (default: %(default)s)') 

    
    args = parser.parse_args()
    
    return args


def main(args: argparse.Namespace):
    if args.model_path != '':
        datetime_now = args.model_path
    else:
        datetime_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join('.', 'save', str(args.model).lower(), datetime_now)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    args.save_path = model_save_path
        
    logger = MainLogger(args)
    logger.debug(f'args: {vars(args)}')

    logger.debug(f'init data preprocessing')
    
    if args.model == ModelEnum.DCGAN:
        data_pre = preprocess.dcgan_data_preprocessor.DCGANDataPreprocessor(args)
        data_pre.transform_data()
        
        model_g = DCGAN.Generator()
        model_d = DCGAN.Discriminator()
        trainer = train.dcgan_trainer.DCGANTrainer(args, model_g, model_d, data_pre)
    elif args.model == ModelEnum.CGAN:
        data_pre = preprocess.cgan_data_preprocessor.CGANDataPreprocessor(args)
        data_pre.transform_data()
        
        model_g = CGAN.Generator()
        model_d = CGAN.Discriminator()
        trainer = train.cgan_trainer.CGANTrainer(args, model_g, model_d, data_pre)
    elif args.model == ModelEnum.DDPM:
        train.ddpm_trainer.train(args)
    elif args.model == ModelEnum.DDIM:
        trainer = train.ddim_trainer.DDIMTrainer(args)
        if args.test:
            trainer.test()
        else:
            trainer.train()
    elif args.model == ModelEnum.BIGGAN:
        trainer = train.biggan_trainer.BIGGANTrainer(args)
        if args.test:
            trainer.test()
        else:
            trainer.train()
    
    # model = QsingBertModel()
    # trainer = Trainer(args, model, data_prep)
    # trainer.train()


if __name__ == "__main__":
    args = get_arg_parse()
    
    main(args)
