import numpy as np
import math
import functools

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn import Parameter as P

from model.BIGGAN import layers


class Generator(nn.Module):
  def __init__(self, skip_init=False):
    super(Generator, self).__init__()
    
    self.dim_z = 128
    self.bottom_width = 4
    self.is_attention = [0]
    self.G_shared = False
    self.shared_dim = 0 if 0 > 0 else self.dim_z
    self.hier = False
    self.cross_replica = False
    self.mybn = False
    # 1e-12 실험
    self.SN_eps = 1e-8
    
    self.in_channels = [256, 256, 256]
    self.out_channels = [256, 256, 256]
    self.upsample = [True, True, True]
    self.resolution = [8, 16, 32]
    self.attention = {2**i: (2**i in [item for item in self.is_attention]) for i in range(3,6)}

    self.num_slots = 1
    self.z_chunk_size = 0

    self.which_conv = functools.partial(layers.SNConv2d,
                        kernel_size=3, padding=1,
                        num_svs=1, num_itrs=1,
                        eps=self.SN_eps)
    self.which_linear = functools.partial(layers.SNLinear,
                        num_svs=1, num_itrs=1,
                        eps=self.SN_eps)
    self.activation = nn.ReLU(inplace=False)
      
    self.which_embedding = nn.Embedding
    bn_linear = (functools.partial(self.which_linear, bias=False) if self.G_shared
                 else self.which_embedding)
    self.which_bn = functools.partial(layers.ccbn,
                          which_linear=bn_linear,
                          cross_replica=self.cross_replica,
                          mybn=self.mybn,
                          input_size=(self.shared_dim + self.z_chunk_size if self.G_shared
                                      else 100))

    self.shared = (self.which_embedding(100, self.shared_dim) if self.G_shared 
                    else layers.identity())
    self.linear = self.which_linear(self.dim_z // self.num_slots,
                                    self.in_channels[0] * (self.bottom_width **2))

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

    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
    
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

  
  def forward(self, z, y):
    if self.hier:
      zs = torch.split(z, self.z_chunk_size, 1)
      z = zs[0]
      ys = [torch.cat([y, item], 1) for item in zs[1:]]
    else:
      ys = [y] * len(self.blocks)
      
    h = self.linear(z)
    h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)
    
    for index, blocklist in enumerate(self.blocks):
      for block in blocklist:
        h = block(h, ys[index])
    return torch.tanh(self.output_layer(h))


class Discriminator(nn.Module):

  def __init__(self, skip_init=False):
    super(Discriminator, self).__init__()
    self.is_attention = [0]
    # 1e-12 실험
    self.SN_eps = 1e-8
    
    self.in_channels = [3, 256, 256, 256]
    self.out_channels = [256, 256, 256, 256]
    self.downsample = [True, True, False, False]
    self.resolution = [16, 16, 16, 16]
    self.attention = {2**i: 2**i in [item for item in self.is_attention] for i in range(2,6)}
    
    self.activation = nn.ReLU(inplace=False)

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
    
    self.blocks = []
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
      if self.attention[self.resolution[index]]:
        self.blocks[-1] += [layers.Attention(self.out_channels[index], self.which_conv)]
    
    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
    self.linear = self.which_linear(self.out_channels[-1], 1)
    self.embed = self.which_embedding(100, self.out_channels[-1])

    if not skip_init:
      self.init_weights()


  def init_weights(self):
    self.param_count = 0
    for module in self.modules():
      if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Embedding)):
        init.normal_(module.weight, 0, 0.02)
        self.param_count += sum([p.data.nelement() for p in module.parameters()])
    print('Param count for D''s initialized parameters: %d' % self.param_count)


  def forward(self, x, y=None):
    h = x
    for index, blocklist in enumerate(self.blocks):
      for block in blocklist:
        h = block(h)
    h = torch.sum(self.activation(h), [2, 3])
    
    out = self.linear(h)
    out = out + torch.sum(self.embed(y) * h, 1, keepdim=True)
    
    return out


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
    
    D_out = self.D(D_input, D_class)
    if x is not None:
      return torch.split(D_out, [G_z.shape[0], x.shape[0]])
    else:
      return D_out
