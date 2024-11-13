import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import models
import torchvision.transforms.functional as F

import os
import pickle

import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import entropy

from utils import get_default_device
from tqdm import tqdm


class Metrics:
    def __init__(self, real_images):
        self.device = get_default_device()
        
        self.inception_model = models.inception_v3()
        self.inception_model.aux_logits = False
        self.inception_model.fc = nn.Sequential(
            nn.Linear(self.inception_model.fc.in_features, 100)
        )
        self.inception_model.load_state_dict(torch.load(os.path.join('./save/iception_v3', f'loss_bset.pt')))
        self.inception_model.to(self.device)
        
        save_path = os.path.join('./data', 'metric_data.pikl')
        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                self.real_features = pickle.load(f)
        else:
            self.real_features = self.__extract_features(real_images, real=True)
            with open(save_path, 'wb') as f:
                pickle.dump(self.real_features, f, pickle.HIGHEST_PROTOCOL)
        
        
    def __extract_features(self, images, real=False, softmax=False):
        self.inception_model.eval()
        features = []
        for image in images:
            if real: image = image[0]
            image = image.to(self.device)
            with torch.no_grad():
                feature = self.inception_model(image)
            
            if softmax:
                out = nn.functional.softmax(feature, dim=1).detach().cpu().numpy()
            else:
                out = feature.cpu().numpy()
            features.append(out)
        return np.vstack(features)


    def inception_score(self, images, splits=10):
        n = len(images.dataset)
        preds = self.__extract_features(images, softmax=True)

        split_scores = []
        for k in range(splits):
            part = preds[k * (n // splits): (k + 1) * (n // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))
        return np.mean(split_scores)


    def fid(self, generated_images):
        generated_features = self.__extract_features(generated_images)
        
        mu1, sigma1 = np.mean(self.real_features, axis=0), np.cov(self.real_features, rowvar=False)
        mu2, sigma2 = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)
        
        diff = np.sum((mu1 - mu2) ** 2.0)
        
        covmean = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        return diff + np.trace(sigma1 + sigma2 - 2.0 * covmean)


    def intra_fid(self, generated_images):
        split_scores = 0
        for class_name in range(0, 100, 10):
            fid_score = self.fid(torch.utils.data.DataLoader(generated_images[class_name * 10:(class_name + 1) * 10], batch_size=128))
            split_scores += fid_score
        return split_scores / 100
