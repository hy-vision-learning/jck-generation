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
        '--num_epochs', type=int, default=100,
        help='Number of epochs to train for (default: %(default)s)')
    parser.add_argument(
        '--accumulate_stats', action='store_true', default=False,
        help='Accumulate "standing" batchnorm stats? (default: %(default)s)')
    parser.add_argument(
        '--num_standing_accumulations', type=int, default=16,
        help='Number of forward passes to use in accumulating standing stats? '
            '(default: %(default)s)')        
    
    parser.add_argument(
        '--save_every', type=int, default=2000,
        help='Save every X iterations (default: %(default)s)')
    parser.add_argument(
        '--num_best_copies', type=int, default=2,
        help='How many previous best checkpoints to save (default: %(default)s)')
    parser.add_argument(
        '--which_best', type=str, default='IS',
        help='Which metric to use to determine when to save new "best"'
            'checkpoints, one of IS or FID (default: %(default)s)')
    parser.add_argument(
        '--test_every', type=int, default=5000,
        help='Test every X iterations (default: %(default)s)')
    parser.add_argument(
        '--num_inception_images', type=int, default=50000,
        help='Number of samples to compute inception metrics with '
            '(default: %(default)s)')
    parser.add_argument(
        '--data_root', type=str, default='data',
        help='Default location where data is stored (default: %(default)s)')
    parser.add_argument(
        '--weights_root', type=str, default='weights',
        help='Default location to store weights (default: %(default)s)')
            
    ### EMA Stuff ###
    parser.add_argument(
        '--ema_decay', type=float, default=0.9999,
        help='EMA decay rate (default: %(default)s)')
    parser.add_argument(
        '--ema_start', type=int, default=0,
        help='When to start updating the EMA weights (default: %(default)s)')
    
    ### Ortho reg stuff ### 
    parser.add_argument(
        '--G_ortho', type=float, default=0.0,
        help='Modified ortho reg coefficient in G')
    parser.add_argument(
        '--D_ortho', type=float, default=0.0,
        help='Modified ortho reg coefficient in D')

    
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
        args.model = 'biggan'
        if args.test:
            trainer.test()
        else:
            trainer.run(vars(args))
    
    # model = QsingBertModel()
    # trainer = Trainer(args, model, data_prep)
    # trainer.train()


if __name__ == "__main__":
    args = get_arg_parse()
    
    main(args)
