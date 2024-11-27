import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import models
from torch.nn import functional as F

import os
import pickle

import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import entropy

from utils import get_default_device
from tqdm import tqdm

from pytorch_fid.fid_score import calculate_frechet_distance
from model.inception_v3 import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d

import inceptionID
from logger.main_logger import MainLogger


class Metrics:
    def __init__(self, args):
        self.args = args
        self.logger = MainLogger(args)
        
        self.data_mu = np.load('./data/cifar100_inception_moments.npz')['mu']
        self.data_sigma = np.load('./data/cifar100_inception_moments.npz')['sigma']
        self.super_mu = np.load('./data/cifar100_inception_moments.npz')['super_mu']
        self.super_sigma = np.load('./data/cifar100_inception_moments.npz')['super_sigma']
        
        self.pool = np.load('./data/cifar100_inception_moments.npz')['pool']
        self.logits = np.load('./data/cifar100_inception_moments.npz')['logits']
        self.labels = np.load('./data/cifar100_inception_moments.npz')['labels']
        
        self.net = self.load_inception_net()
    
    
    def load_inception_net(self):
        return inceptionID.load_inception_net()
    
    
    def get_inception_metrics(self, sample, num_inception_images, num_splits=10, use_torch=True, print_intra=False):
        self.logger.debug(f'use_torch: {use_torch}\tgenerating samples')
        g_pool, g_logits, g_labels = inceptionID.accumulate_inception_activations(sample, self.net, num_inception_images, batch_size=self.args.batch_size)
        
        self.logger.debug('Calculating Inception Score')
        IS_mean, IS_std = inceptionID.calculate_inception_score(g_logits.cpu().numpy(), num_splits)
        
        self.logger.debug('Calculating means and covariances')
        if use_torch:
            mu, sigma = torch.mean(g_pool, 0), inceptionID.torch_cov(g_pool, rowvar=False)
        else:
            mu, sigma = np.mean(g_pool.cpu().numpy(), axis=0), np.cov(g_pool.cpu().numpy(), rowvar=False)
        
        self.logger.debug('Calculating FID')
        if use_torch:
            FID = inceptionID.torch_calculate_fid(mu, sigma,
                    torch.tensor(self.data_mu).float().cuda(), torch.tensor(self.data_sigma).float().cuda(), atol=1e-8)
            FID = float(FID.cpu().numpy())
        else:
            FID = inceptionID.calculate_fid(mu, sigma, self.data_mu, self.data_sigma)
        
        intra_fids = None
        if print_intra:
            self.logger.debug('Calculating intra-FID')
            intra_fids, _ = inceptionID.calculate_intra_fid(self.super_mu, self.super_sigma, g_pool, g_logits, g_labels, chage_superclass=True, use_torch=False)
        
        del mu, sigma, g_pool, g_logits, g_labels
        
        self.logger.debug(f'Inception Score: {IS_mean} +/- {IS_std}, FID: {FID}, ' + (f'Intra-FID: {intra_fids}' if print_intra else ''))
        return IS_mean, IS_std, FID, intra_fids
