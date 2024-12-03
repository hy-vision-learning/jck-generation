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
    
    parser.add_argument('--batch_size', type=int, help='배치 크기', default=50)
    parser.add_argument('--num_epochs', type=int, help='에포크 수', default=500)
    parser.add_argument('--num_D_steps', type=int, help='Discriminator 단계 수', default=4)
    parser.add_argument('--G_ortho', type=float, help='Generator의 Ortho 값', default=0.0)
    parser.add_argument('--D_ortho', type=float, help='Discriminator의 Ortho 값', default=0.0)
    parser.add_argument('--ema_start', type=int, help='EMA 시작 시점', default=1000)
    parser.add_argument('--test_every', type=int, help='테스트 주기', default=500)
    parser.add_argument('--full_test_counter', type=int, help='5만장 전체 테스트 주기 (테스트 n번마다 한번)', default=0)
    parser.add_argument('--save_every', type=int, help='저장 주기', default=2000)
    parser.add_argument('--ema_decay', type=float, help='EMA 감쇠율', default=0.999)
    parser.add_argument('--num_inception_images', type=int, help='인셉션 메트릭 계산 샘플 수', default=10000)
    parser.add_argument('--superclass', type=int, help='lable superclass 적용 여부', default=1)
    parser.add_argument('--dict_size', type=int, help='dict size', default=10)
    parser.add_argument('--commitment', type=float, help='commitment', default=1.0)
    parser.add_argument('--dict_decay', type=float, help='dict decay', default=0.8)
    # parser.add_argument('--quantization', type=int, help='quantization 적용 여부', default=0)
    
    args = parser.parse_args()
    
    args.superclass = args.superclass == 1
    # args.quantization = args.quantization == 1
    
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
    logger.debug(f'args: {'\n'.join([f"{key}: {value}" for key, value in vars(args).items()])}')

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
