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
    
    parser.add_argument('-w', '--num_worker', type=int, help='DataLoader worker', default=0)
    parser.add_argument('-b', '--batch_size', type=int, help='학습 배치사이즈', default=128)
    
    parser.add_argument('-e', '--epoch', type=int, help='epoch', default=100)
    parser.add_argument('-mlr', '--max_learning_rate', type=float, help='optimizer max learning rate 설정', default=0.1)
    parser.add_argument('-milr', '--min_learning_rate', type=float, help='optimizer min learning rate 설정', default=1e-4)
    parser.add_argument('-wd', '--weight_decay', type=float, help='optimizer weight decay 설정', default=5e-4)
    parser.add_argument('-snt', '--nesterov', type=int, help="nesterov sgd 사용 여부", default=1)
    
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
    elif args.model == ModelEnum.DDIM:
        data_pre = preprocess.ddim_preprocessor.DDIMDataPreprocessor(args)
        data_pre.transform_data()
        
        trainer = train.ddim_trainer.DDIMTrainer(args, data_pre)
    
    # model = QsingBertModel()
    # trainer = Trainer(args, model, data_prep)
    trainer.train()


if __name__ == "__main__":
    args = get_arg_parse()
    
    main(args)
