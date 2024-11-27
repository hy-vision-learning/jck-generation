import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import inceptionID
import torch

import os

norm_mean = [0.5,0.5,0.5]
norm_std = [0.5,0.5,0.5]
image_size = 32,32


train_transform = []
train_transform = transforms.Compose(train_transform + [
                     transforms.ToTensor(),
                        transforms.Resize((299, 299)),
                     transforms.Normalize(norm_mean, norm_std)])


train_dataset = torchvision.datasets.CIFAR100(
    root="./data",  # 데이터 저장 경로
    train=True,     # 학습용 데이터셋
    download=True,  # 데이터셋 다운로드
    transform=train_transform
)

train_loader = DataLoader(train_dataset, batch_size=64,
                              shuffle=True)
net = inceptionID.load_inception_net()

pool, logits, labels = inceptionID.get_net_output(device="cuda:0", train_loader=train_loader, net=net)
mu_data, sigma_data = np.mean(pool, axis=0), np.cov(pool, rowvar=False)



super_class = inceptionID.super_class_mapping()
super_labels = [super_class[i] for i in labels]
super_labels = np.array(super_labels)

super_mu = np.zeros((100, 2048), dtype=np.float32)
super_sigma = np.zeros((100, 2048, 2048), dtype=np.float32)
for super_idx, _ in inceptionID.superclass_mapping.items():
    mask = (super_labels == super_idx)
    pool_low = pool[mask]
    mu_part, sigma_part = np.mean(pool_low, axis=0), np.cov(pool_low, rowvar=False)
    super_mu[super_idx] = mu_part
    super_sigma[super_idx] = sigma_part


if not os.path.exists('./data'):
    os.makedirs('./data')
np.savez('./data/cifar100_inception_moments.npz', **{'mu' : mu_data, 'sigma' : sigma_data,
                                                     'super_mu' : super_mu, 'super_sigma' : super_sigma,
                                                     'pool': pool, 'logits': logits, 'labels': labels})
