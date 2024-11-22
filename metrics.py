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
        
        # self.inception_model = models.inception_v3(pretrained=True, transform_input=False).to(self.device)
        self.inception_model = InceptionV3([3, 4]).to(self.device)
        self.inception_model.eval()
        # self.inception_model.aux_logits = False
        # self.inception_model.fc = nn.Sequential(
        #     nn.Linear(self.inception_model.fc.in_features, 100)
        # )
        # self.inception_model.load_state_dict(torch.load(os.path.join('./save/iception_v3', f'loss_bset.pt')))
        # self.inception_model.to(self.device)
        
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
                torch.utils.data.DataLoader(
                    real_images, 128, shuffle=False, num_workers=0, pin_memory=True), real=True
                )[0].cpu().numpy()
            with open(save_path, 'wb') as f:
                pickle.dump(self.real_features, f, pickle.HIGHEST_PROTOCOL)
        
        
    # def __extract_features(self, images, real=False):
    #     self.inception_model.eval()
    #     features = []
    #     for image in images:
    #         if real: image = image[0]
    #         image = image.to(self.device)
    #         with torch.no_grad():
    #             feature = self.inception_model(image)[0]
    #         out = feature.cpu().numpy()
    #         features.append(out)
    #     return np.vstack(features)
        
        
    def __extract_features(self, images, real=False):
        self.inception_model.eval()
        features_fid = []
        features_prob = []
        for image in images:
            if real: image = image[0]
            image = image.to(self.device)
            with torch.no_grad():
                fearue = self.inception_model(image)
            # if fearue.size(2) != 1 or fearue.size(3) != 1:
            #     fearue = adaptive_avg_pool2d(fearue, output_size=(1, 1))
            features_fid.append(fearue[0])
            features_prob.append(fearue[1])
        return torch.vstack(features_fid).squeeze(), torch.vstack(features_prob)


    def inception_score(self, images, splits=10):
        n = len(images.dataset)
        preds = self.__extract_features(images)[1].cpu().numpy()

        scores = []
        for i in range(splits):
            part = preds[
                (i * n // splits):
                ((i + 1) * n // splits), :]
            kl = part * (
                np.log(part) -
                np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
        return np.mean(scores), np.std(scores)


    def fid(self, generated_images, intra_fid=False, label=0, eps=1e-6):
        """
        Frechet Inception Distance (FID)을 계산하는 메서드입니다.
        
        Args:
            generated_images (torch.Tensor): 생성된 이미지 텐서.
            intra_fid (bool, optional): 클래스 내에서 FID를 계산할지 여부. 기본값은 False.
            label (int, optional): intra_fid가 True일 때 사용할 클래스 라벨. 기본값은 0.
            eps (float, optional): 안정성을 위한 작은 상수. 기본값은 1e-6.
        
        Returns:
            float: 계산된 FID 값.
        """
        # 생성된 이미지에서 특성 추출
        generated_features = self.__extract_features(generated_images)[0].cpu().numpy()
        real_features = self.real_features
        
        if intra_fid:
            # 클래스 내 FID 계산을 위해 특정 클래스에 해당하는 실제 특성의 평균과 공분산 계산
            mu1 = np.mean(real_features[self.real_superclass_idx[label]], axis=0)
            sigma1 = np.cov(real_features[self.real_superclass_idx[label]], rowvar=False)
        else:
            # 전체 데이터에 대한 실제 특성의 평균과 공분산 계산
            mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        
        # 생성된 이미지의 특성에 대한 평균과 공분산 계산
        mu2, sigma2 = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)
        
        # 평균값의 차이 계산
        diff = mu1 - mu2
        # 공분산 행렬의 중간 값 계산을 위해 sqrtm 함수 사용
        covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
        
        # 공분산 행렬이 유한한지 확인
        if not np.isfinite(covmean).all():
            # 공분산 행렬에 작은 오프셋을 추가하여 안정성 확보
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # 공분산 행렬이 복소수인지 확인
        if np.iscomplexobj(covmean):
            # 허수 부가 거의 없는지 확인
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            # 허수 부가 거의 없으면 실수 부분만 추출
            covmean = covmean.real

        # 공분산 행렬의 트레이스 계산
        tr_covmean = np.trace(covmean)
        
        # FID 계산 공식 적용: diff.dot(diff) + trace(sigma1) + trace(sigma2) - 2 * trace(covmean)
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


    def intra_fid(self, generated_images):
        split_scores = 0
        for sidx in range(20):
            fid_score = self.fid(
                torch.utils.data.DataLoader(
                    generated_images[self.fake_superclass_idx[sidx]],
                    128, pin_memory=True, num_workers=0, shuffle=False),
                intra_fid=True, label=sidx)
            split_scores += fid_score
        return split_scores / 100
