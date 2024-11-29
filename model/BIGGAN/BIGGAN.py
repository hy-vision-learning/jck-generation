import numpy as np
import math
import functools

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn import Parameter as P

from model.BIGGAN import layers


class Quantize(nn.Module):
	def __init__(self, dim, n_embed, commitment=1.0, decay=0.8, eps=1e-5):
		super().__init__()

		self.dim = dim
		self.n_embed = n_embed
		self.decay = decay
		self.eps = eps
		self.commitment = commitment

		embed = torch.randn(dim, n_embed)
		self.register_buffer('embed', embed)
		self.register_buffer('cluster_size', torch.zeros(n_embed))
		self.register_buffer('embed_avg', embed.clone())

	def forward(self, x, y=None):
		x = x.permute(0, 2, 3, 1).contiguous()
		input_shape = x.shape
		flatten = x.reshape(-1, self.dim)
		dist = (
		    flatten.pow(2).sum(1, keepdim=True)
		    - 2 * flatten @ self.embed
		    + self.embed.pow(2).sum(0, keepdim=True)
		)
		_, embed_ind = (-dist).max(1)
		embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
		embed_ind = embed_ind.view(*x.shape[:-1])
		quantize = self.embed_code(embed_ind).view(input_shape)

		if self.training:
			self.cluster_size.data.mul_(self.decay).add_(
			    1 - self.decay, embed_onehot.sum(0)
			)
			embed_sum = flatten.transpose(0, 1) @ embed_onehot
			self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum)
			n = self.cluster_size.sum()
			cluster_size = (
			    (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
			)
			embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
			self.embed.data.copy_(embed_normalized)

		diff = self.commitment*torch.mean(torch.mean((quantize.detach() - x).pow(2), dim=(1,2)),
		                                  dim=(1,), keepdim=True)
		quantize = x + (quantize - x).detach()
		avg_probs = torch.mean(embed_onehot, 0)
		perplexity = torch.exp(- torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

		return quantize.permute(0, 3, 1, 2).contiguous(), diff, perplexity

	def embed_code(self, embed_id):
		return F.embedding(embed_id, self.embed.transpose(0, 1))


class Generator(nn.Module):
    """
    BigGAN의 생성기 클래스.

    이 클래스는 BigGAN에서 생성기 모델의 아키텍처를 정의합니다.
    선택적인 어텐션 레이어가 있는 여러 개의 잔여 블록(GBlock)으로 구성되며,
    마지막에 최종 이미지를 생성하는 출력 레이어가 있습니다.

    속성:
        dim_z (int): 잠재 공간의 차원 (노이즈 벡터).
        bottom_width (int): 첫 번째 선형 레이어 후의 초기 공간 차원.
        is_attention (list): 특정 해상도에서 어텐션을 적용할지 여부를 나타내는 리스트.
        G_shared (bool): 공유 임베딩을 사용할지 여부를 결정하는 플래그.
        shared_dim (int): 공유 임베딩의 차원.
        hier (bool): 계층적 잠재 공간을 활성화할지 여부를 나타내는 플래그.
        cross_replica (bool): 크로스 레플리카 배치 정규화를 활성화할지 여부를 나타내는 플래그.
        mybn (bool): 사용자 정의 배치 정규화를 활성화할지 여부를 나타내는 플래그.
        SN_eps (float): 스펙트럴 정규화의 epsilon 값.
        in_channels (list): 각 GBlock의 입력 채널 수 리스트.
        out_channels (list): 각 GBlock의 출력 채널 수 리스트.
        upsample (list): 각 GBlock에서 업샘플링을 적용할지 여부를 나타내는 리스트.
        resolution (list): 각 GBlock에 해당하는 해상도 리스트.
        attention (dict): 특정 해상도에서 어텐션 레이어를 적용할지 여부를 나타내는 딕셔너리.
        num_slots (int): 계층적 잠재 공간의 슬롯 수.
        z_chunk_size (int): 계층적 잠재 공간에서 각 청크의 크기.
        which_conv (function): 스펙트럴 정규화가 적용된 컨볼루션 레이어.
        which_linear (function): 스펙트럴 정규화가 적용된 선형 레이어.
        activation (nn.Module): 블록에서 사용되는 활성화 함수.
        which_embedding (nn.Module): 임베딩 레이어.
        which_bn (function): 배치 정규화 레이어.
        shared (nn.Module): 공유 임베딩 레이어 또는 항등 함수.
        linear (nn.Module): 잠재 벡터를 변환하는 첫 번째 선형 레이어.
        blocks (nn.ModuleList): GBlocks 및 어텐션 레이어의 리스트.
        output_layer (nn.Sequential): 이미지를 생성하는 최종 출력 레이어.
    """
    def __init__(self, skip_init=False, n_classes=100):
        super(Generator, self).__init__()
        
        # 잠재 공간 및 아키텍처 파라미터
        self.dim_z = 128
        self.bottom_width = 4
        self.is_attention = [0]
        self.G_shared = False
        self.shared_dim = self.dim_z
        self.hier = False
        self.cross_replica = False
        self.mybn = False
        # 1e-12 실험
        self.SN_eps = 1e-8
        self.n_classes = n_classes
        
        # GBlocks를 위한 채널 구성
        self.in_channels = [256, 256, 256]
        self.out_channels = [256, 256, 256]
        self.upsample = [True, True, True]
        self.resolution = [8, 16, 32]
        # 어텐션을 적용할 위치 결정
        self.attention = {2**i: (2**i in [item for item in self.is_attention]) for i in range(3,6)}

        self.num_slots = 1
        self.z_chunk_size = 0

        # 스펙트럴 정규화가 적용된 컨볼루션 및 선형 레이어 정의
        self.which_conv = functools.partial(layers.SNConv2d,
                            kernel_size=3, padding=1,
                            num_svs=1, num_itrs=1,
                            eps=self.SN_eps)
        self.which_linear = functools.partial(layers.SNLinear,
                            num_svs=1, num_itrs=1,
                            eps=self.SN_eps)
        self.activation = nn.ReLU(inplace=False)
        
        # 임베딩 및 배치 정규화 레이어 정의
        self.which_embedding = nn.Embedding
        bn_linear = (functools.partial(self.which_linear, bias=False) if self.G_shared
                     else self.which_embedding)
        self.which_bn = functools.partial(layers.ccbn,
                              which_linear=bn_linear,
                              cross_replica=self.cross_replica,
                              mybn=self.mybn,
                              input_size=(self.shared_dim + self.z_chunk_size if self.G_shared else self.n_classes))

        # 공유 임베딩 또는 항등 함수
        self.shared = (self.which_embedding(self.n_classes, self.shared_dim) if self.G_shared 
                        else layers.identity())
        # 잠재 벡터를 피처 맵으로 변환하는 첫 번째 선형 레이어
        self.linear = self.which_linear(self.dim_z // self.num_slots,
                                        self.in_channels[0] * (self.bottom_width **2))

        # 선택적인 어텐션과 함께 GBlocks 정의
        self.blocks = []
        for index in range(len(self.out_channels)):
            self.blocks += [[layers.GBlock(in_channels=self.in_channels[index],
                                 out_channels=self.out_channels[index],
                                 which_conv=self.which_conv,
                                 which_bn=self.which_bn,
                                 activation=self.activation,
                                 upsample=(functools.partial(F.interpolate, scale_factor=2)
                                           if self.upsample[index] else None))]]
            if self.attention[self.resolution[index]]:
                self.blocks[-1] += [layers.Attention(self.out_channels[index], self.which_conv)]

        # 적절한 등록을 위해 blocks를 ModuleList로 변환
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
        
        # 최종 출력 레이어 정의: BatchNorm -> ReLU -> Conv
        self.output_layer = nn.Sequential(layers.bn(self.out_channels[-1],
                                                    cross_replica=self.cross_replica,
                                                    mybn=self.mybn),
                                        self.activation,
                                        self.which_conv(self.out_channels[-1], 3))

        if not skip_init:
            self.init_weights()

    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d) 
                or isinstance(module, nn.Linear) 
                or isinstance(module, nn.Embedding)):
                init.normal_(module.weight, 0, 0.02)
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print(f'Param count for G\'s initialized parameters: {self.param_count}')

    def forward(self, z, y):
        if self.hier:
            # 계층적 생성을 위한 잠재 벡터 분할
            zs = torch.split(z, self.z_chunk_size, 1)
            z = zs[0]
            ys = [torch.cat([y, item], 1) for item in zs[1:]]
        else:
            # 각 블록에 대해 레이블 텐서를 반복
            ys = [y] * len(self.blocks)
        
        # 잠재 벡터를 초기 피처 맵으로 변환
        h = self.linear(z)
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)
        
        # GBlocks 및 어텐션 레이어를 통과
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h, ys[index])
        return torch.tanh(self.output_layer(h))


class Discriminator(nn.Module):
    """
    BigGAN의 판별기 클래스.

    이 클래스는 BigGAN에서 판별기 모델의 아키텍처를 정의합니다.
    선택적인 어텐션 레이어가 있는 여러 개의 잔여 블록(DBlock)으로 구성되며,
    마지막에 실제/가짜 점수를 출력하는 선형 레이어가 있습니다.

    속성:
        is_attention (list): 특정 해상도에서 어텐션을 적용할지 여부를 나타내는 리스트.
        SN_eps (float): 스펙트럴 정규화의 epsilon 값.
        in_channels (list): 각 DBlock의 입력 채널 수 리스트.
        out_channels (list): 각 DBlock의 출력 채널 수 리스트.
        downsample (list): 각 DBlock에서 다운샘플링을 적용할지 여부를 나타내는 리스트.
        resolution (list): 각 DBlock에 해당하는 해상도 리스트.
        attention (dict): 특정 해상도에서 어텐션 레이어를 적용할지 여부를 나타내는 딕셔너리.
        activation (nn.Module): 블록에서 사용되는 활성화 함수.
        which_conv (function): 스펙트럴 정규화가 적용된 컨볼루션 레이어.
        which_linear (function): 스펙트럴 정규화가 적용된 선형 레이어.
        which_embedding (function): 스펙트럴 정규화가 적용된 임베딩 레이어.
        blocks (nn.ModuleList): DBlocks 및 어텐션 레이어의 리스트.
        linear (nn.Module): 판별기 점수를 생성하는 최종 선형 레이어.
        embed (nn.Module): 프로젝션 기반 판별을 위한 임베딩 레이어.
    """
  
    def __init__(self, skip_init=False, n_classes=100,
                 dict_size=10, dict_decay=0.8, commitment=0.5):
        super(Discriminator, self).__init__()
        self.is_attention = [0]
        # 1e-12 실험
        self.SN_eps = 1e-8
        self.n_classes = n_classes
        
        # DBlocks를 위한 채널 구성
        self.in_channels = [3, 256, 256, 256]
        self.out_channels = [256, 256, 256, 256]
        self.downsample = [True, True, False, False]
        self.resolution = [16, 16, 16, 16]
        # 어텐션을 적용할 위치 결정
        self.attention = {2**i: 2**i in [item for item in self.is_attention] for i in range(2,6)}
        
        self.activation = nn.ReLU(inplace=False)

        # 스펙트럴 정규화가 적용된 컨볼루션 및 선형 레이어 정의
        self.which_conv = functools.partial(layers.SNConv2d,
                            kernel_size=3, padding=1,
                            num_svs=1, num_itrs=1,
                            eps=self.SN_eps)
        self.which_linear = functools.partial(layers.SNLinear,
                            num_svs=1, num_itrs=1,
                            eps=self.SN_eps)
        self.which_embedding = functools.partial(layers.SNEmbedding,
                                num_svs=1, num_itrs=1,
                                eps=self.SN_eps)
        
        # 선택적인 어텐션과 함께 DBlocks 정의
        self.blocks = []
        self.quant_layer = [0, 1, 2, 3]
        for index in range(len(self.out_channels)):
            self.blocks += [[layers.DBlock(
                            in_channels=self.in_channels[index],
                            out_channels=self.out_channels[index],
                            which_conv=self.which_conv,
                            wide=True,
                            activation=self.activation,
                            preactivation=(index > 0),
                            downsample=(nn.AvgPool2d(2) if self.downsample[index] else None))
                          ]]
            if index in self.quant_layer:
                self.blocks[-1] += [Quantize(self.out_channels[index], 2 ** dict_size, commitment=commitment, decay=dict_decay)]
            if self.attention[self.resolution[index]]:
                self.blocks[-1] += [layers.Attention(self.out_channels[index], self.which_conv)]
        
        # 적절한 등록을 위해 blocks를 ModuleList로 변환
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
        # 판별기 점수를 생성하는 최종 선형 레이어
        self.linear = self.which_linear(self.out_channels[-1], 1)
        # 프로젝션 기반 판별을 위한 임베딩 레이어
        self.embed = self.which_embedding(self.n_classes, self.out_channels[-1])

        if not skip_init:
            self.init_weights()


    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Embedding)):
                init.normal_(module.weight, 0, 0.02)
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for D\'s initialized parameters: %d' % self.param_count)


    def forward(self, x, y=None):
        h = x
        quant_loss = 0
        # DBlocks 및 어텐션 레이어를 통과
        for index, blocklist in enumerate(self.blocks):
            if index in self.quant_layer:
                h = blocklist[0](h)
                h_, diff, ppl = blocklist[1](h)
                if len(blocklist) == 3:
                    h = blocklist[2](h)
                quant_loss += diff
            else:
                for block in blocklist:
                    h = block(h)
        # 글로벌 합 풀링
        h = torch.sum(self.activation(h), [2, 3])
        
        out = self.linear(h)
        # 레이블이 제공된 경우 프로젝션 점수 추가
        out = out + torch.sum(self.embed(y) * h, 1, keepdim=True)
        
        return out, quant_loss, ppl


class BigGAN(nn.Module):
    def __init__(self, G, D):
        super(BigGAN, self).__init__()
        self.G = G
        self.D = D


    def forward(self, z, gy, x=None, dy=None, train_G=False):              
        with torch.set_grad_enabled(train_G):
            G_z = self.G(z, self.G.shared(gy))
        
        D_input = torch.cat([G_z, x], 0) if x is not None else G_z
        D_class = torch.cat([gy, dy], 0) if dy is not None else gy
        
        D_out, quant_loss, ppl = self.D(D_input, D_class)
        if x is not None:
            D_real, D_fake = torch.split(D_out, [G_z.shape[0], x.shape[0]])
            quant_loss_real, quant_loss_fake = torch.split(quant_loss, (G_z.shape[0], x.shape[0]), dim=0)
            return D_real, D_fake, quant_loss_real, quant_loss_fake, ppl.view(-1, 1)
        else:
            return D_out, quant_loss