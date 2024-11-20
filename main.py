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
    
    # parser.add_argument('-w', '--num_worker', type=int, help='DataLoader worker', default=0)
    # parser.add_argument('-b', '--batch_size', type=int, help='학습 배치사이즈', default=128)
    
    # parser.add_argument('-e', '--epoch', type=int, help='epoch', default=100)
    # parser.add_argument('-mlr', '--max_learning_rate', type=float, help='optimizer max learning rate 설정', default=0.1)
    # parser.add_argument('-milr', '--min_learning_rate', type=float, help='optimizer min learning rate 설정', default=1e-4)
    parser.add_argument('-wd', '--weight_decay', type=float, help='optimizer weight decay 설정', default=0.0)
    # parser.add_argument('-snt', '--nesterov', type=int, help="nesterov sgd 사용 여부", default=1)
    
    # Unet
    parser.add_argument('--ch', type=int, help='base channel of UNet', default=128)
    parser.add_argument('--ch_mult', nargs='+', type=int, help='channel multiplier', default=[1, 2, 2, 2])
    parser.add_argument('--attn', nargs='+', type=int, help='add attention to these levels', default=[1])
    parser.add_argument('--num_res_blocks', type=int, help='# resblock in each level', default=2)
    parser.add_argument('--dropout', type=float, help='dropout rate of resblock', default=0.1)

    # Gaussian Diffusion
    parser.add_argument('--beta_1', type=float, help='start beta value', default=1e-4)
    parser.add_argument('--beta_T', type=float, help='end beta value', default=0.02)
    parser.add_argument('--T', type=int, help='total diffusion steps', default=1000)
    parser.add_argument('--mean_type', type=str, choices=['xprev', 'xstart', 'epsilon'], help='predict variable', default='epsilon')
    parser.add_argument('--var_type', type=str, choices=['fixedlarge', 'fixedsmall'], help='variance type', default='fixedlarge')

    # Training
    parser.add_argument('--lr', type=float, help='target learning rate', default=2e-4)
    parser.add_argument('--grad_clip', type=float, help="gradient norm clipping", default=-1.0)
    parser.add_argument('--total_steps', type=int, help='total training steps', default=800000)
    parser.add_argument('--img_size', type=int, help='image size', default=32)
    parser.add_argument('--warmup', type=int, help='learning rate warmup', default=5000)
    parser.add_argument('--batch_size', type=int, help='batch size', default=128)
    parser.add_argument('--num_workers', type=int, help='workers of Dataloader', default=4)
    parser.add_argument('--ema_decay', type=float, help="ema decay rate", default=0.9999)
    parser.add_argument('--parallel', type=bool, help='multi gpu training', default=False)

    # Logging & Sampling
    # parser.add_argument('--logdir', type=str, help='log directory', default='./logs/DDPM_CIFAR10_EPS')
    parser.add_argument('--sample_size', type=int, help="sampling size of images", default=64)
    parser.add_argument('--sample_step', type=int, help='frequency of sampling', default=1000)
    parser.add_argument('--eta', type=float, help='sampler eta', default=0.0)

    # Evaluation
    parser.add_argument('--save_step', type=int, help='frequency of saving checkpoints, 0 to disable during training', default=5000)
    parser.add_argument('--eval_step', type=int, help='frequency of evaluating model, 0 to disable during training', default=0)
    parser.add_argument('--num_images', type=int, help='the number of generated images for evaluation', default=50000)
    parser.add_argument('--eval_sample_step', type=int, help='frequency of sampling', default=500)
    # parser.add_argument('--fid_use_torch', type=bool, help='calculate IS and FID on gpu', default=False)
    # parser.add_argument('--fid_cache', type=str, help='FID cache', default='./stats/cifar10.train.npz')

    
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
        trainer.train()
    
    # model = QsingBertModel()
    # trainer = Trainer(args, model, data_prep)
    # trainer.train()


if __name__ == "__main__":
    args = get_arg_parse()
    
    main(args)
