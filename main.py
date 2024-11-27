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
    
    parser.add_argument('--test', type=int, help='테스트모드', default=0)
    parser.add_argument('--model_path', type=str, help='모델 폴더 이름', default='')
    
    parser.add_argument('--log_file', type=int, help='로그 파일 출력 여부. 0=false, 1=true', default=1)
    
    parser.add_argument('--model', type=ModelEnum, help='학습 모델', choices=list(ModelEnum), default=ModelEnum.DCGAN)
    
    parser.add_argument('--num_workers', type=int, help='DataLoader workers', default=4)
    
    args = parser.parse_args()
    
    args.batch_size = 50
    args.num_epochs = 500
    args.num_D_steps = 4
    args.G_ortho = 0.0
    args.ema_start = 1000
    args.test_every = 500
    args.save_every = 2000
    
    args.ema_decay = 0.999
    args.toggle_grads = True
    args.split_D = False
    args.D_ortho = 0.0
    args.G_ortho = 0.0
    args.num_inception_images = 10000
    
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
    
    print(f'args.model: {args.model}')
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
            trainer.run()


if __name__ == "__main__":
    args = get_arg_parse()
    
    main(args)
