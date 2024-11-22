from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from utils import get_default_device

def extract(v: torch.Tensor, t: torch.LongTensor, shape: Tuple[int, ...]) -> torch.Tensor:
    """
    주어진 텐서 `v`에서 타임스텝 `t`에 해당하는 값을 추출하고, 
    필요한 형상으로 확장합니다.
    """
    out = torch.gather(v, index=t, dim=0)
    out = out.to(device=t.device, dtype=torch.float32)
    out = out.view([t.shape[0]] + [1] * (len(shape) - 1))
    return out


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class DDIMForwardTrainer(nn.Module):
    def __init__(self, model: nn.Module, beta_1: float, beta_T: float, T: int):
        """
        DDIM Forward Trainer 초기화

        Args:
            model (nn.Module): Unet 모델
            beta (Tuple[float, float]): 노이즈 스케줄의 시작과 끝 값
            T (int): 총 타임스텝 수
        """
        super(DDIMForwardTrainer, self).__init__()
        self.model = model
        self.T = T

        # 노이즈 스케줄 생성
        # self.register_buffer("beta_t", torch.linspace(beta_1, beta_T, T, dtype=torch.float32))
        self.register_buffer("beta_t", cosine_beta_schedule(T))
        alpha_t = 1.0 - self.beta_t
        alpha_t_bar = torch.cumprod(alpha_t, dim=0)

        # 신호 및 노이즈 비율 계산
        self.register_buffer("signal_rate", torch.sqrt(alpha_t_bar))
        self.register_buffer("noise_rate", torch.sqrt(1.0 - alpha_t_bar))

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Forward 과정: 노이즈 추가 및 예측된 노이즈의 손실 반환

        Args:
            x0 (torch.Tensor): 원본 이미지 텐서 (배치 크기, 채널, 높이, 너비)

        Returns:
            torch.Tensor: 계산된 손실 값
        """
        self.model.train()
        
        # 랜덤 타임스텝 샘플링
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)

        # 노이즈 샘플링
        epsilon = torch.randn_like(x0)

        # 현재 타임스텝의 신호 및 노이즈 비율 추출
        signal = extract(self.signal_rate, t, x0.shape)
        noise = extract(self.noise_rate, t, x0.shape)

        # 노이즈가 추가된 이미지 생성
        x_t = signal * x0 + noise * epsilon

        # Unet 모델을 통해 노이즈 예측
        epsilon_theta = self.model(x_t, t)

        # MSE 손실 계산
        loss = F.mse_loss(epsilon_theta, epsilon)
        # loss = (epsilon - epsilon_theta).square().sum(dim=(1, 2, 3)).mean(dim=0)
        # loss = torch.sum(loss)
        return loss


class DDIMSampler(nn.Module):
    def __init__(self, model, beta_1: float, beta_T: float, T: int):
        """
        DDIMSampler 초기화
        
        Args:
            model (nn.Module): 노이즈를 예측하는 모델 (예: Unet)
            beta (Tuple[float, float]): 노이즈 스케줄의 시작과 끝 값
            T (int): 총 타임스텝 수
        """
        super().__init__()
        self.model = model  # 노이즈 예측 모델을 저장
        self.T = T  # 총 타임스텝 수를 저장

        # 베타 스케줄 생성: 주어진 범위에서 T개의 베타 값을 균등하게 생성
        beta_t = torch.linspace(beta_1, beta_T, T, dtype=torch.float32)
        
        # 알파 값 계산: alpha_t = 1 - beta_t
        alpha_t = 1.0 - beta_t
        
        # 누적 알파 값 계산 (Paper의 $\bar{\alpha_t}$에 해당) 및 버퍼에 등록
        self.register_buffer("alpha_t_bar", torch.cumprod(alpha_t, dim=0))

    @torch.no_grad()
    def sample_one_step(self, x_t, time_step: int, prev_time_step: int, eta: float):
        """
        한 단계 샘플링을 수행하여 이전 타임스텝의 이미지를 생성
        
        Args:
            x_t (torch.Tensor): 현재 타임스텝의 이미지 텐서
            time_step (int): 현재 타임스텝 인덱스
            prev_time_step (int): 이전 타임스텝 인덱스
            eta (float): 논문에서의 sigma 파라미터 계수. eta=0이면 DDIM, eta=1이면 DDPM
        
        Returns:
            torch.Tensor: 이전 타임스텝의 이미지 텐서 x_{t-1}
        """
        # 현재 및 이전 타임스텝을 나타내는 텐서 생성
        t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)
        prev_t = torch.full((x_t.shape[0],), prev_time_step, device=x_t.device, dtype=torch.long)
        
        # 현재 및 이전 타임스텝의 누적 알파 값 추출
        alpha_t = extract(self.alpha_t_bar, t, x_t.shape)
        alpha_t_prev = extract(self.alpha_t_bar, prev_t, x_t.shape)
        
        # 모델을 사용하여 현재 타임스텝의 노이즈 예측
        self.model.eval()
        epsilon_theta_t = self.model(x_t, t)
        
        # x_{t-1} 계산을 위한 sigma_t 계산
        sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
        
        # 표준 가우시안 노이즈 생성
        epsilon_t = torch.randn_like(x_t)
        
        # x_{t-1} 계산 공식 적용
        x_t_minus_one = (
            torch.sqrt(alpha_t_prev / alpha_t) * x_t +
            (torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) - torch.sqrt((alpha_t_prev * (1 - alpha_t)) / alpha_t)) * epsilon_theta_t +
            sigma_t * epsilon_t
        )
        return x_t_minus_one

    @torch.no_grad()
    def forward(self, x_t, steps: int = 1, method="linear", eta=0.0,
                only_return_x_0: bool = True, interval: int = 1):
        """
        샘플링 과정을 통해 최종 이미지를 생성
        
        Parameters:
            x_t (torch.Tensor): 표준 가우시안 노이즈. 형태는 (배치 크기, 채널, 높이, 너비).
            steps (int): 샘플링 단계 수.
            method (str): 샘플링 방법, "linear" 또는 "quadratic" 가능.
            eta (float): 논문에서의 sigma 파라미터 계수. eta=0은 DDIM, eta=1은 DDPM을 의미.
            only_return_x_0 (bool): 샘플링 과정 중 이미지를 저장할지 여부. True이면 최종 결과만 반환.
            interval (int): `only_return_x_0=False`일 때만 유효. 중간 과정을 저장할 간격을 결정.
        
        Returns:
            torch.Tensor: 
                - `only_return_x_0=True`이면 형태는 (배치 크기, 채널, 높이, 너비)
                - `only_return_x_0=False`이면 형태는 (배치 크기, 샘플 수, 채널, 높이, 너비)로 중간 이미지 포함
        """
        # 샘플링 방법에 따라 타임스텝 시퀀스 생성
        if method == "linear":
            a = self.T // steps
            time_steps = np.asarray(list(range(0, self.T, a)))
        elif method == "quadratic":
            time_steps = (np.linspace(0, np.sqrt(self.T * 0.8), steps) ** 2).astype(np.int)
        
        # 샘플링 과정에서의 최종 alpha 값을 맞추기 위해 1을 더함
        time_steps = time_steps + 1
        # 이전 타임스텝 시퀀스 생성 (첫 번째는 0으로 설정)
        time_steps_prev = np.concatenate([[0], time_steps[:-1]])
        
        x = [x_t]  # 샘플링 과정 중 저장할 이미지 리스트 초기화
        # 샘플링 과정을 진행하며 tqdm으로 진행 상태 표시
        for i in reversed(range(0, steps)):
            # 한 단계 샘플링 수행하여 이전 타임스텝의 이미지 생성
            x_t = self.sample_one_step(x_t, time_steps[i], time_steps_prev[i], eta)
            
            # 중간 결과 저장 조건 확인
            if not only_return_x_0 and ((steps - i) % interval == 0 or i == 0):
                # 이미지 클리핑: 값의 범위를 [-1.0, 1.0]으로 제한
                x.append(torch.clamp(x_t, -1.0, 1.0))
        
        # only_return_x_0 옵션에 따라 결과 반환
        if only_return_x_0:
            return x_t  # 최종 이미지 반환: [배치 크기, 채널, 높이, 너비]
        return torch.stack(x, dim=1)  # 중간 이미지 포함: [배치 크기, 샘플 수, 채널, 높이, 너비]


class EMAHelper(object):
    def __init__(self, mu=0.999, device=None):
        """
        EMAHelper 클래스의 초기화 메서드.
        
        Args:
            mu (float): EMA 업데이트를 위한 모멘텀 계수. 일반적으로 0.999가 사용됩니다.
        """
        self.mu = mu  # EMA 업데이트에 사용되는 모멘텀 계수
        self.shadow = {}  # EMA로 업데이트된 파라미터를 저장할 딕셔너리
        self.device = device
        if self.device is None:
            self.device = get_default_device()

    def register(self, module):
        """
        모델의 파라미터를 EMA 추적에 등록합니다.
        
        Args:
            module (nn.Module): EMA를 적용할 PyTorch 모델 또는 모듈.
        """
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()  # 파라미터를 클론하여 shadow에 저장

    def update(self, module):
        """
        현재 모델의 파라미터를 사용하여 shadow 파라미터를 EMA 방식으로 업데이트합니다.
        
        Args:
            module (nn.Module): EMA를 업데이트할 PyTorch 모델 또는 모듈.
        """
        for name, param in module.named_parameters():
            if param.requires_grad:
                # EMA 업데이트 공식: shadow = mu * shadow + (1 - mu) * param
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        """
        shadow에 저장된 EMA 파라미터를 현재 모델에 복사하여 적용합니다.
        
        Args:
            module (nn.Module): EMA를 적용할 PyTorch 모델 또는 모듈.
        """
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)  # shadow 파라미터를 현재 모델의 파라미터에 복사

    def ema_copy(self, module):
        """
        EMA가 적용된 모델의 복사본을 생성하여 반환합니다.
        
        Args:
            module (nn.Module): EMA가 적용될 PyTorch 모델 또는 모듈.
        
        Returns:
            nn.Module: EMA 파라미터가 적용된 모델의 복사본.
        """
        module_copy = type(module)(**module.config).to(self.device)
        module_copy.load_state_dict(module.state_dict())  # 원본 모듈의 상태를 복사
        # module_copy = copy.deepcopy(module)  # deepcopy 대신 수동 복사 사용
        self.ema(module_copy)  # shadow 파라미터를 복사된 모듈에 적용
        return module_copy  # EMA가 적용된 모델 복사본 반환

    def state_dict(self):
        """
        현재 shadow 파라미터의 상태를 반환합니다.
        
        Returns:
            dict: shadow 파라미터의 상태.
        """
        return self.shadow

    def load_state_dict(self, state_dict):
        """
        저장된 shadow 파라미터 상태를 로드하여 복원합니다.
        
        Args:
            state_dict (dict): 저장된 shadow 파라미터 상태.
        """
        self.shadow = state_dict
    
    