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
        
        self.class_to_superclass = {
            4: 0, 30: 0, 55: 0, 72: 0, 95: 0,
            1: 1, 32: 1, 67: 1, 73: 1, 91: 1, 
            54: 2, 62: 2, 70: 2, 82: 2, 92: 2, 
            9: 3, 10: 3, 16: 3, 28: 3, 61: 3, 
            0: 4, 51: 4, 53: 4, 57: 4, 83: 4,
            22: 5, 39: 5, 40: 5, 86: 5, 87: 5,
            5: 6, 20: 6, 25: 6, 84: 6, 94: 6,
            6: 7, 7: 7, 14: 7, 18: 7, 24: 7,
            3: 8, 42: 8, 43: 8, 88: 8, 97: 8,
            12: 9, 17: 9, 37: 9, 68: 9, 76: 9,
            23: 10, 33: 10, 49: 10, 60: 10, 71: 10,
            15: 11, 19: 11, 21: 11, 31: 11, 38: 11,
            34: 12, 63: 12, 64: 12, 66: 12, 75: 12,
            26: 13, 45: 13, 77: 13, 79: 13, 99: 13,
            2: 14, 11: 14, 35: 14, 46: 14, 98: 14,
            27: 15, 29: 15, 44: 15, 78: 15, 93: 15,
            36: 16, 50: 16, 65: 16, 74: 16, 80: 16,
            47: 17, 52: 17, 56: 17, 59: 17, 96: 17,
            8: 18, 13: 18, 48: 18, 58: 18, 90: 18,
            41: 19, 69: 19, 81: 19, 85: 19, 89: 19
        }
        
        self.inception_model = models.inception_v3()
        self.inception_model.aux_logits = False
        self.inception_model.fc = nn.Sequential(
            nn.Linear(self.inception_model.fc.in_features, 100)
        )
        self.inception_model.load_state_dict(torch.load(os.path.join('./save/iception_v3', f'loss_bset.pt')))
        self.inception_model.to(self.device)
        
        save_path = os.path.join('./data', 'metric_data.pikl')
        
        real_targets = real_images.targets
        fake_targets = []
        for i in range(100):
            fake_targets.extend([i] * 10)
        
        self.real_superclass_idx = dict()
        self.fake_superclass_idx = dict()
        for sidx in range(20):
            idx = [i for i, t in enumerate(real_targets) if self.class_to_superclass[t] == sidx]
            self.real_superclass_idx[sidx] = idx
            
            idx = [i for i, t in enumerate(fake_targets) if self.class_to_superclass[t] == sidx]
            self.fake_superclass_idx[sidx] = idx
        
        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                self.real_features = pickle.load(f)
        else:
            self.real_features = self.__extract_features(
                torch.utils.data.DataLoader(real_images, 128, shuffle=False, num_workers=0, pin_memory=True), real=True)
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


    def fid(self, generated_images, intra_fid=False, label=0):
        generated_features = self.__extract_features(generated_images)
        
        if intra_fid:
            mu1 = np.mean(self.real_features[self.real_superclass_idx[label]], axis=0)
            sigma1 = np.cov(self.real_features[self.real_superclass_idx[label]], rowvar=False)
        else:
            mu1, sigma1 = np.mean(self.real_features, axis=0), np.cov(self.real_features, rowvar=False)
        mu2, sigma2 = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)
        
        diff = np.sum((mu1 - mu2) ** 2.0)
        
        covmean = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        return diff + np.trace(sigma1 + sigma2 - 2.0 * covmean)


    def intra_fid(self, generated_images):
        split_scores = 0
        for sidx in range(20):
            fid_score = self.fid(
                torch.utils.data.DataLoader(generated_images[self.fake_superclass_idx[sidx]], 128, pin_memory=True, num_workers=0, shuffle=False),
                intra_fid=True, label=sidx)
            split_scores += fid_score
        return split_scores / 100
