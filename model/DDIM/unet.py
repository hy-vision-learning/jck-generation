import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from abc import ABC, abstractmethod


class TimestepBlock(nn.Module):
    """
    타임스텝 정보를 입력으로 받는 기본 블록 클래스.
    모든 타임스텝 관련 블록은 이 클래스를 상속받아야 합니다.
    """
    @abstractmethod
    def forward(self, x, emb):
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    nn.Sequential을 확장하여 타임스텝 임베딩을 처리할 수 있는 시퀀셜 블록.
    각 레이어가 TimestepBlock 인스턴스인지를 확인하고, 타임스텝 정보를 전달합니다.
    """
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


def norm_layer(channels):
    """
    그룹 정규화 레이어를 생성하는 헬퍼 함수.
    
    Args:
        channels (int): 입력 채널 수.
    
    Returns:
        nn.GroupNorm: 그룹 정규화 레이어.
    """
    return nn.GroupNorm(32, channels)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResidualBlock(TimestepBlock):
    """
    타임스텝 임베딩을 통합한 잔차 블록.
    입력 특징과 타임스텝 임베딩을 결합하여 깊이 있는 특징을 학습합니다.
    """
    def __init__(self, in_channels, out_channels, time_channels, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            norm_layer(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )

        self.conv2 = nn.Sequential(
            norm_layer(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        h = self.conv1(x)
        h += self.time_emb(t)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """
    공간적 주의를 적용하는 어텐션 블록.
    입력 특징에 대한 쿼리, 키, 밸류를 계산하고 어텐션을 적용하여 중요한 특징을 강조합니다.
    """
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0

        self.norm = norm_layer(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))  # 쿼리, 키, 밸류 계산
        q, k, v = qkv.reshape(B * self.num_heads, -1, H * W).chunk(3, dim=1)
        scale = 1. / math.sqrt(math.sqrt(C // self.num_heads))
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)  # 어텐션 스코어 계산
        attn = attn.softmax(dim=-1)  # 소프트맥스 정규화
        h = torch.einsum("bts,bcs->bct", attn, v)  # 어텐션 적용
        h = h.reshape(B, -1, H, W)
        h = self.proj(h)  # 출력 변환
        return h + x  # 잔차 연결


class Upsample(nn.Module):
    """
    업샘플링 레이어.
    이미지의 해상도를 두 배로 증가시키고, 필요 시 합성곱을 적용하여 특징을 보강합니다.
    """
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")  # 업샘플링
        if self.use_conv:
            x = self.conv(x)  # 합성곱 적용
        return x


class Downsample(nn.Module):
    """
    다운샘플링 레이어.
    이미지의 해상도를 절반으로 줄이고, 필요 시 합성곱을 적용하여 특징을 변환합니다.
    """
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(stride=2, kernel_size=2)

    def forward(self, x):
        return self.op(x)


class UNet(nn.Module):
    """
    UNet 아키텍처 구현.
    DDIM에서 노이즈를 예측하기 위해 인코더-디코더 구조와 어텐션 메커니즘을 사용합니다.
    """
    def __init__(
            self,
            in_channels=3,
            model_channels=128,
            out_channels=3,
            num_res_blocks=2,
            attention_resolutions=(8, 16),
            dropout=0,
            channel_mult=(1, 2, 2, 2),
            conv_resample=True,
            num_heads=4
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        
        self.config = {
            'in_channels': in_channels,
            'model_channels': model_channels,
            'out_channels': out_channels,
            'num_res_blocks': num_res_blocks,
            'attention_resolutions': attention_resolutions,
            'dropout': dropout,
            'channel_mult': channel_mult,
            'conv_resample': conv_resample,
            'num_heads': num_heads
        }

        # 타임스텝 임베딩 레이어 설정
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        # self.time_embed = nn.Sequential(
        #     nn.Linear(model_channels, time_embed_dim),
        #     nn.SiLU(),
        #     nn.Linear(time_embed_dim, time_embed_dim),
        # )

        # 다운샘플링 블록 초기화
        self.down_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1))
        ])
        down_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResidualBlock(ch, mult * model_channels, time_embed_dim, dropout)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                down_block_chans.append(ch)
            if level != len(channel_mult) - 1:  # 마지막 스테이지는 다운샘플링하지 않음
                self.down_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample)))
                down_block_chans.append(ch)
                ds *= 2

        # 중간 블록 초기화
        self.middle_block = TimestepEmbedSequential(
            ResidualBlock(ch, ch, time_embed_dim, dropout),
            AttentionBlock(ch, num_heads=num_heads),
            ResidualBlock(ch, ch, time_embed_dim, dropout)
        )

        # 업샘플링 블록 초기화
        self.up_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResidualBlock(
                        ch + down_block_chans.pop(),
                        model_channels * mult,
                        time_embed_dim,
                        dropout
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.up_blocks.append(TimestepEmbedSequential(*layers))

        # 출력 레이어 설정
        self.out = nn.Sequential(
            norm_layer(ch),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
        )
        
    # def timestep_embedding(self, timesteps, dim, max_period=10000):
    #     """
    #     타임스텝을 고차원 임베딩 벡터로 변환합니다.
    #     주기적 함수(사인 및 코사인)를 사용하여 타임스텝 정보를 인코딩합니다.
        
    #     Args:
    #         timesteps (torch.Tensor): 타임스텝 값.
    #         dim (int): 임베딩 차원.
    #         max_period (int): 주기 범위.
        
    #     Returns:
    #         torch.Tensor: 타임스텝 임베딩 벡터.
    #     """
    #     half = dim // 2
    #     freqs = torch.exp(
    #         -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    #     ).to(device=timesteps.device)
    #     args = timesteps[:, None].float() * freqs[None]
    #     embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    #     if dim % 2:
    #         embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    #     return embedding

    def forward(self, x, timesteps):
        """
        UNet의 순전파 과정.
        인코더를 통해 특징을 추출하고, 중간 블록에서 처리한 후 디코더를 통해 이미지를 복원합니다.
        
        Args:
            x (torch.Tensor): 입력 이미지 텐서 (배치 크기, 채널, 높이, 너비).
            timesteps (torch.Tensor): 타임스텝 값.
        
        Returns:
            torch.Tensor: 출력 이미지 텐서.
        """
        hs = []
        # emb = self.time_embed(self.timestep_embedding(timesteps, self.model_channels))
        emb = self.time_embed(timesteps)

        h = x
        # 인코더 단계
        for module in self.down_blocks:
            h = module(h, emb)
            hs.append(h)
        # 중간 블록
        h = self.middle_block(h, emb)
        
        # 디코더 단계
        for module in self.up_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
            
        # 출력 생성
        return self.out(h)