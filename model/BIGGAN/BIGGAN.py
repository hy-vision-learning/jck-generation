import numpy as np
import math
import functools

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P

from model.BIGGAN import layers
from model.BIGGAN.sync_batchnorm import SynchronizedBatchNorm2d as SyncBatchNorm2d


class Generator(nn.Module):
  def __init__(self, skip_init=False, no_optim=False, **kwargs):
    super(Generator, self).__init__()
    # Channel width mulitplier
    self.ch = 64
    # Dimensionality of the latent space
    self.dim_z = 128
    # The initial spatial dimensions
    self.bottom_width = 4
    # Resolution of the output
    self.resolution = 32
    # Kernel size?
    self.kernel_size = 3
    # Attention?
    self.attention = '0'
    # number of classes, for use in categorical conditional generation
    self.n_classes = 100
    # Use shared embeddings?
    self.G_shared = False
    # Dimensionality of the shared embedding? Unused if not using G_shared
    self.shared_dim = 0 if 0 > 0 else self.dim_z
    # Hierarchical latent space?
    self.hier = False
    # Cross replica batchnorm?
    self.cross_replica = False
    # Use my batchnorm?
    self.mybn = False
    # nonlinearity for residual blocks
    self.activation = nn.ReLU(inplace=False)
    # Initialization style
    self.init = 'N02'
    # Parameterization style
    self.G_param = 'SN'
    # Normalization style
    self.norm_style = 'bn'
    # Epsilon for BatchNorm?
    self.BN_eps = 1e-5
    # Epsilon for Spectral Norm? 1e-12
    self.SN_eps = 1e-8
    # Architecture dict
    self.arch = {
      'in_channels' :  [self.ch * item for item in [4, 4, 4]],
      'out_channels' : [self.ch * item for item in [4, 4, 4]],
      'upsample' : [True] * 3,
      'resolution' : [8, 16, 32],
      'attention' : {2**i: (2**i in [int(item) for item in self.attention.split('_')]) for i in range(3,6)}
    }

    self.num_slots = 1
    self.z_chunk_size = 0

    self.which_conv = functools.partial(layers.SNConv2d,
                        kernel_size=3, padding=1,
                        num_svs=1, num_itrs=1,
                        eps=self.SN_eps)
    self.which_linear = functools.partial(layers.SNLinear,
                        num_svs=1, num_itrs=1,
                        eps=self.SN_eps)
      
    # We use a non-spectral-normed embedding here regardless;
    # For some reason applying SN to G's embedding seems to randomly cripple G
    self.which_embedding = nn.Embedding
    bn_linear = (functools.partial(self.which_linear, bias=False) if self.G_shared
                 else self.which_embedding)
    self.which_bn = functools.partial(layers.ccbn,
                          which_linear=bn_linear,
                          cross_replica=self.cross_replica,
                          mybn=self.mybn,
                          input_size=(self.shared_dim + self.z_chunk_size if self.G_shared
                                      else self.n_classes),
                          norm_style=self.norm_style,
                          eps=self.BN_eps)


    # Prepare model
    # If not using shared embeddings, self.shared is just a passthrough
    self.shared = (self.which_embedding(self.n_classes, self.shared_dim) if self.G_shared 
                    else layers.identity())
    # First linear layer
    self.linear = self.which_linear(self.dim_z // self.num_slots,
                                    self.arch['in_channels'][0] * (self.bottom_width **2))

    # self.blocks is a doubly-nested list of modules, the outer loop intended
    # to be over blocks at a given resolution (resblocks and/or self-attention)
    # while the inner loop is over a given block
    self.blocks = []
    for index in range(len(self.arch['out_channels'])):
      self.blocks += [[layers.GBlock(in_channels=self.arch['in_channels'][index],
                             out_channels=self.arch['out_channels'][index],
                             which_conv=self.which_conv,
                             which_bn=self.which_bn,
                             activation=self.activation,
                             upsample=(functools.partial(F.interpolate, scale_factor=2)
                                       if self.arch['upsample'][index] else None))]]

      # If attention on this block, attach it to the end
      if self.arch['attention'][self.arch['resolution'][index]]:
        print('Adding attention layer in G at resolution %d' % self.arch['resolution'][index])
        self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]

    # Turn self.blocks into a ModuleList so that it's all properly registered.
    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

    # output layer: batchnorm-relu-conv.
    # Consider using a non-spectral conv here
    self.output_layer = nn.Sequential(layers.bn(self.arch['out_channels'][-1],
                                                cross_replica=self.cross_replica,
                                                mybn=self.mybn),
                                    self.activation,
                                    self.which_conv(self.arch['out_channels'][-1], 3))

    # Initialize weights. Optionally skip init for testing.
    if not skip_init:
      self.init_weights()

    # Set up optimizer
    # If this is an EMA copy, no need for an optim, so just return now
    if no_optim:
      return
    self.lr, self.B1, self.B2, self.adam_eps = 2e-4, 0.0, 0.999, 1e-8
    self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                          betas=(self.B1, self.B2), weight_decay=0,
                          eps=self.adam_eps)

    # LR scheduling, left here for forward compatibility
    # self.lr_sched = {'itr' : 0}# if self.progressive else {}
    # self.j = 0

  # Initialize
  def init_weights(self):
    self.param_count = 0
    for module in self.modules():
      if (isinstance(module, nn.Conv2d) 
          or isinstance(module, nn.Linear) 
          or isinstance(module, nn.Embedding)):
        if self.init == 'ortho':
          init.orthogonal_(module.weight)
        elif self.init == 'N02':
          init.normal_(module.weight, 0, 0.02)
        elif self.init in ['glorot', 'xavier']:
          init.xavier_uniform_(module.weight)
        else:
          print('Init style not recognized...')
        self.param_count += sum([p.data.nelement() for p in module.parameters()])
    print('Param count for G''s initialized parameters: %d' % self.param_count)

  # Note on this forward function: we pass in a y vector which has
  # already been passed through G.shared to enable easy class-wise
  # interpolation later. If we passed in the one-hot and then ran it through
  # G.shared in this forward function, it would be harder to handle.
  def forward(self, z, y):
    # If hierarchical, concatenate zs and ys
    if self.hier:
      zs = torch.split(z, self.z_chunk_size, 1)
      z = zs[0]
      ys = [torch.cat([y, item], 1) for item in zs[1:]]
    else:
      ys = [y] * len(self.blocks)
      
    # First linear layer
    h = self.linear(z)
    # Reshape
    h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)
    
    # Loop over blocks
    for index, blocklist in enumerate(self.blocks):
      # Second inner loop in case block has multiple layers
      for block in blocklist:
        h = block(h, ys[index])
        
    # Apply batchnorm-relu-conv-tanh at output
    return torch.tanh(self.output_layer(h))


class Discriminator(nn.Module):

  def __init__(self, skip_init=False, **kwargs):
    super(Discriminator, self).__init__()
    # Width multiplier
    self.ch = 64
    # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
    self.D_wide = True
    # Resolution
    self.resolution = 32
    # Kernel size
    self.kernel_size = 3
    # Attention?
    self.attention = '0'
    # Number of classes
    self.n_classes = 100
    # Activation
    self.activation = nn.ReLU(inplace=False)
    # Initialization style
    self.init = 'N02'
    # Parameterization style
    self.D_param = 'SN'
    # Epsilon for Spectral Norm? 1e-12
    self.SN_eps = 1e-8
    # Architecture
    self.arch =  {
      'in_channels' :  [3] + [item * self.ch for item in [4, 4, 4]],
      'out_channels' : [item * self.ch for item in [4, 4, 4, 4]],
      'downsample' : [True, True, False, False],
      'resolution' : [16, 16, 16, 16],
      'attention' : {2**i: 2**i in [int(item) for item in self.attention.split('_')] for i in range(2,6)}
    }

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
    # Prepare model
    # self.blocks is a doubly-nested list of modules, the outer loop intended
    # to be over blocks at a given resolution (resblocks and/or self-attention)
    self.blocks = []
    for index in range(len(self.arch['out_channels'])):
      self.blocks += [[layers.DBlock(in_channels=self.arch['in_channels'][index],
                       out_channels=self.arch['out_channels'][index],
                       which_conv=self.which_conv,
                       wide=self.D_wide,
                       activation=self.activation,
                       preactivation=(index > 0),
                       downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]
      # If attention on this block, attach it to the end
      if self.arch['attention'][self.arch['resolution'][index]]:
        print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
        self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index],
                                             self.which_conv)]
    # Turn self.blocks into a ModuleList so that it's all properly registered.
    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
    # Linear output layer. The output dimension is typically 1, but may be
    # larger if we're e.g. turning this into a VAE with an inference output
    self.linear = self.which_linear(self.arch['out_channels'][-1], 1)
    # Embedding for projection discrimination
    self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])

    # Initialize weights
    if not skip_init:
      self.init_weights()

    # Set up optimizer
    self.lr, self.B1, self.B2, self.adam_eps = 2e-4, 0.0, 0.999, 1e-8
      
    self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                            betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
    # LR scheduling, left here for forward compatibility
    # self.lr_sched = {'itr' : 0}# if self.progressive else {}
    # self.j = 0

  # Initialize
  def init_weights(self):
    self.param_count = 0
    for module in self.modules():
      if (isinstance(module, nn.Conv2d)
          or isinstance(module, nn.Linear)
          or isinstance(module, nn.Embedding)):
        if self.init == 'ortho':
          init.orthogonal_(module.weight)
        elif self.init == 'N02':
          init.normal_(module.weight, 0, 0.02)
        elif self.init in ['glorot', 'xavier']:
          init.xavier_uniform_(module.weight)
        else:
          print('Init style not recognized...')
        self.param_count += sum([p.data.nelement() for p in module.parameters()])
    print('Param count for D''s initialized parameters: %d' % self.param_count)

  def forward(self, x, y=None):
    # Stick x into h for cleaner for loops without flow control
    h = x
    # Loop over blocks
    for index, blocklist in enumerate(self.blocks):
      for block in blocklist:
        h = block(h)
    # Apply global sum pooling as in SN-GAN
    h = torch.sum(self.activation(h), [2, 3])
    # Get initial class-unconditional output
    out = self.linear(h)
    # Get projection of final featureset onto class vectors and add to evidence
    out = out + torch.sum(self.embed(y) * h, 1, keepdim=True)
    return out

# Parallelized G_D to minimize cross-gpu communication
# Without this, Generator outputs would get all-gathered and then rebroadcast.
class G_D(nn.Module):
  def __init__(self, G, D):
    super(G_D, self).__init__()
    self.G = G
    self.D = D

  def forward(self, z, gy, x=None, dy=None, train_G=False, return_G_z=False,
              split_D=False):              
    # If training G, enable grad tape
    with torch.set_grad_enabled(train_G):
      # Get Generator output given noise
      G_z = self.G(z, self.G.shared(gy))
      # Cast as necessary
    # Split_D means to run D once with real data and once with fake,
    # rather than concatenating along the batch dimension.
    if split_D:
      D_fake = self.D(G_z, gy)
      if x is not None:
        D_real = self.D(x, dy)
        return D_fake, D_real
      else:
        if return_G_z:
          return D_fake, G_z
        else:
          return D_fake
    # If real data is provided, concatenate it with the Generator's output
    # along the batch dimension for improved efficiency.
    else:
      D_input = torch.cat([G_z, x], 0) if x is not None else G_z
      D_class = torch.cat([gy, dy], 0) if dy is not None else gy
      # Get Discriminator output
      D_out = self.D(D_input, D_class)
      if x is not None:
        return torch.split(D_out, [G_z.shape[0], x.shape[0]]) # D_fake, D_real
      else:
        if return_G_z:
          return D_out, G_z
        else:
          return D_out
