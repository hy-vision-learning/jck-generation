import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter as P
from torchvision.models.inception import inception_v3
import numpy as np
from scipy.linalg import sqrtm

superclass_mapping = {
    0: [4, 30, 55, 72, 95],  # aquatic mammals
    1: [1, 32, 67, 73, 91],  # fish
    2: [54, 62, 70, 82, 92],  # flowers
    3: [9, 10, 16, 28, 61],  # food containers
    4: [0, 51, 53, 57, 83],  # fruit and vegetables
    5: [22, 39, 40, 86, 87],  # household electrical devices
    6: [5, 20, 25, 84, 94],  # household furniture
    7: [6, 7, 14, 18, 24],  # insects
    8: [3, 42, 43, 88, 97],  # large carnivores
    9: [12, 17, 37, 68, 76],  # large man-made outdoor things
    10: [23, 33, 49, 60, 71],  # large natural outdoor scenes
    11: [15, 19, 21, 31, 38],  # large omnivores and herbivores
    12: [34, 63, 64, 66, 75],  # medium-sized mammals
    13: [26, 45, 77, 79, 99],  # non-insect invertebrates
    14: [2, 11, 35, 46, 98],  # people
    15: [27, 29, 44, 78, 93],  # reptiles
    16: [36, 50, 65, 74, 80],  # small mammals
    17: [47, 52, 56, 59, 96],  # trees
    18: [8, 13, 48, 58, 90],  # vehicles 1
    19: [41, 69, 81, 85, 89]   # vehicles 2
}


def load_inception_net():
  inception_model = inception_v3(pretrained=True, transform_input=False)
  inception_model = WrapInception(inception_model.eval()).cuda()
  return inception_model


class WrapInception(nn.Module):
  def __init__(self, net):
    super(WrapInception,self).__init__()
    self.net = net
    self.mean = P(torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1),
                  requires_grad=False)
    self.std = P(torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1),
                 requires_grad=False)
  def forward(self, x):
    # Normalize x
    x = (x + 1.) / 2.0
    x = (x - self.mean) / self.std
    # Upsample if necessary
    if x.shape[2] != 299 or x.shape[3] != 299:
      x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)
    # 299 x 299 x 3
    x = self.net.Conv2d_1a_3x3(x)
    # 149 x 149 x 32
    x = self.net.Conv2d_2a_3x3(x)
    # 147 x 147 x 32
    x = self.net.Conv2d_2b_3x3(x)
    # 147 x 147 x 64
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    # 73 x 73 x 64
    x = self.net.Conv2d_3b_1x1(x)
    # 73 x 73 x 80
    x = self.net.Conv2d_4a_3x3(x)
    # 71 x 71 x 192
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    # 35 x 35 x 192
    x = self.net.Mixed_5b(x)
    # 35 x 35 x 256
    x = self.net.Mixed_5c(x)
    # 35 x 35 x 288
    x = self.net.Mixed_5d(x)
    # 35 x 35 x 288
    x = self.net.Mixed_6a(x)
    # 17 x 17 x 768
    x = self.net.Mixed_6b(x)
    # 17 x 17 x 768
    x = self.net.Mixed_6c(x)
    # 17 x 17 x 768
    x = self.net.Mixed_6d(x)
    # 17 x 17 x 768
    x = self.net.Mixed_6e(x)
    # 17 x 17 x 768
    # 17 x 17 x 768
    x = self.net.Mixed_7a(x)
    # 8 x 8 x 1280
    x = self.net.Mixed_7b(x)
    # 8 x 8 x 2048
    x = self.net.Mixed_7c(x)
    # 8 x 8 x 2048
    pool = torch.mean(x.view(x.size(0), x.size(1), -1), 2)
    # 1 x 1 x 2048
    logits = self.net.fc(F.dropout(pool, training=False).view(pool.size(0), -1))
    # 1000 (num_classes)
    return pool, logits
  

def get_net_output(train_loader, net,device):
  pool, logits, labels = [], [], []

  for i, (x, y) in enumerate(train_loader):
      x = x.to(device)
      with torch.no_grad():
        pool_val, logits_val = net(x)
        pool += [np.asarray(pool_val.cpu())]
        logits += [np.asarray(F.softmax(logits_val, 1).cpu())]
        labels += [np.asarray(y.cpu())]
  pool, logits, labels = [np.concatenate(item, 0) for item in [pool, logits, labels]]
  return pool, logits, labels




def accumulate_inception_activations(sample, net, num_inception_images=50000, batch_size=50):
  pool, logits, labels = [], [], []
  
  count = num_inception_images // 100
  assert num_inception_images % 100 == 0, "num_inception_images must be divisible by 100"
  balanced_labels = []
  for i in range(100):
    balanced_labels.extend([i] * count)
  
  i = 0
  while (torch.cat(logits, 0).shape[0] if len(logits) else 0) < num_inception_images:
    with torch.no_grad():
      images, labels_val = sample(set_labels=True, labels=balanced_labels[i:i + batch_size])
      # labels_val = torch.tensor(balanced_labels[i:i + images.shape[0]])
      pool_val, logits_val = net(images.float())
      pool += [pool_val]
      logits += [F.softmax(logits_val, 1)]
      labels += labels_val.tolist()
      del labels_val
      i += batch_size
  return torch.cat(pool, 0), torch.cat(logits, 0), torch.LongTensor(labels)

  
def calculate_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid
  
def sqrt_newton_schulz(A, numIters, dtype=None, epsilon=1e-6):
  with torch.no_grad():
    if dtype is None:
      dtype = A.type()
    batchSize = A.shape[0]
    dim = A.shape[1]
    normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
    Y = A.div(normA.view(batchSize, 1, 1).expand_as(A));
    I = torch.eye(dim,dim).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
    Z = torch.eye(dim,dim).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
    for i in range(numIters):
      # previous_Y = Y.clone()
      T = 0.5*(3.0*I - Z.bmm(Y))
      Y = Y.bmm(T)
      Z = T.bmm(Z)
      # if torch.isnan(Y).any():
      #   Y = previous_Y
      #   break
    sA = Y*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
  return sA

def _approximation_error(matrix: torch.Tensor, s_matrix: torch.Tensor) -> torch.Tensor:
    norm_of_matrix = torch.norm(matrix)
    error = matrix - torch.mm(s_matrix, s_matrix)
    error = torch.norm(error) / norm_of_matrix
    return error


def sqrtm_newton_schulz(matrix: torch.Tensor, num_iters: int=100, atol=1e-5):
    r"""
    Square root of matrix using Newton-Schulz Iterative method
    Source: https://github.com/msubhransu/matrix-sqrt/blob/master/matrix_sqrt.py
    Args:
        matrix: matrix or batch of matrices
        num_iters: Number of iteration of the method
    Returns:
        Square root of matrix
        Error
    """
    expected_num_dims = 2
    if matrix.dim() != expected_num_dims:
        raise ValueError(f'Input dimension equals {matrix.dim()} {matrix.shape}, expected {expected_num_dims}')

    if num_iters <= 0:
        raise ValueError(f'Number of iteration equals {num_iters}, expected greater than 0')

    dim = matrix.size(0)
    norm_of_matrix = matrix.norm(p='fro')
    Y = matrix.div(norm_of_matrix)
    I = torch.eye(dim, dim, requires_grad=False).to(matrix)
    Z = torch.eye(dim, dim, requires_grad=False).to(matrix)

    s_matrix = torch.empty_like(matrix)
    error = torch.empty(1).to(matrix)

    for _ in range(num_iters):
        T = 0.5 * (3.0 * I - Z.mm(Y))
        Y = Y.mm(T)
        Z = T.mm(Z)

        s_matrix = Y * torch.sqrt(norm_of_matrix)
        error = _approximation_error(matrix, s_matrix)
        if torch.isclose(error, torch.tensor([0.]).to(error), atol=atol):
            break

    return s_matrix, error

def torch_cov(m, rowvar=False):
  if m.dim() > 2:
    raise ValueError('m has more than 2 dimensions')
  if m.dim() < 2:
    m = m.view(1, -1)
  if not rowvar and m.size(0) != 1:
    m = m.t()
  fact = 1.0 / (m.size(1) - 1)
  m -= torch.mean(m, dim=1, keepdim=True)
  mt = m.t()
  return fact * m.matmul(mt).squeeze()

def torch_calculate_fid(mu1, sigma1, mu2, sigma2, atol=1e-5):
  diff = mu1 - mu2
  covmean = sqrtm_newton_schulz(sigma1.mm(sigma2), 50, atol)[0]
  # covmean = sqrt_newton_schulz(sigma1.mm(sigma2).unsqueeze(0), 50).squeeze(0)
  # covmean, _ = sqrtm(sigma1.cpu().numpy().dot(sigma2.cpu().numpy()), disp=False)
  # if np.iscomplexobj(covmean):
  #     covmean = covmean.real
  # covmean = torch.from_numpy(covmean).cuda()
  
  out = (diff.dot(diff) +  torch.trace(sigma1) + torch.trace(sigma2) - 2 * torch.trace(covmean))
  return out
  
def calculate_inception_score(pred, num_splits=10):
  scores = []
  for index in range(num_splits):
    pred_chunk = pred[index * (pred.shape[0] // num_splits): (index + 1) * (pred.shape[0] // num_splits), :]
    kl_inception = pred_chunk * (np.log(pred_chunk) - np.log(np.expand_dims(np.mean(pred_chunk, 0), 0)))
    kl_inception = np.mean(np.sum(kl_inception, 1))
    scores.append(np.exp(kl_inception))
  return np.mean(scores), np.std(scores)


def calculate_intra_fid(super_mu, super_sigma, g_pool, g_logits, g_labels, chage_superclass=True):
  # 5만개 원본 데이터의 superclass mean, std는 미리 계산하고 저장한 데이터를 사용하도록 수정
  intra_fids = []
  super_class = super_class_mapping()
  
  # super_labels = [super_class[i] for i in labels]
  # super_labels = np.array(super_labels)
  
  if chage_superclass:
    g_super_labels = [super_class[i] for i in g_labels]
    g_super_labels = np.array(g_super_labels)
  else:
    g_super_labels = np.array(g_labels.cpu())
  
  for super_idx, _ in superclass_mapping.items():
    # mask = (super_labels == super_idx)
    g_mask = (g_super_labels == super_idx)
    
    # pool_low = pool[mask]
    g_pool_low = g_pool[g_mask]
    
    assert 2500 == len(g_pool_low), "super-classes count error"
    # if len(pool_low) == 0 or len(g_pool_low) == 0:
    #   continue
    
    mu, sigma = np.mean(g_pool_low.cpu().numpy(), axis=0), np.cov(g_pool_low.cpu().numpy(), rowvar=False)
    # mu_data, sigma_data = np.mean(pool_low, axis=0), np.cov(pool_low, rowvar=False)
    
    fid = calculate_fid(mu, sigma, super_mu[super_idx], super_sigma[super_idx])
    intra_fids.append(fid)
    
  return np.mean(intra_fids), intra_fids


def torch_calculate_intra_fid(super_mu, super_sigma, g_pool, g_logits, g_labels, chage_superclass=True):
  super_class = super_class_mapping()
  
  # super_labels = [super_class[i] for i in labels]
  # super_labels = np.array(super_labels)
  
  if chage_superclass:
    g_super_labels = [super_class[i] for i in g_labels]
    g_super_labels = np.array(g_super_labels)
  else:
    g_super_labels = np.array(g_labels.cpu())
  
  _, counts = np.unique(g_super_labels, return_counts=True)
  assert np.all(counts == counts[0]), "라벨 개수가 동일하지 않습니다."
  
  # if use_torch:
  #   pool = torch.tensor(pool, device='cuda')
  
  intra_fid = 0
  for super_idx, _ in superclass_mapping.items():
    # mask = (super_labels == super_idx)
    g_mask = (g_super_labels == super_idx)
    
    # pool_low = pool[mask]
    g_pool_low = g_pool[g_mask]
    
    # assert 2500 == len(g_pool_low), "super-classes count error"
    # if len(pool_low) == 0 or len(g_pool_low) == 0:
    #   continue
    
    mu_data, sigma_data = super_mu[super_idx], super_sigma[super_idx]
    # g_pool_low = torch.tensor(g_pool_low, device='cuda')
    mu, sigma = torch.mean(g_pool_low, 0), torch_cov(g_pool_low, rowvar=False)
    mu_data, sigma_data = torch.tensor(mu_data, device='cuda'), torch.tensor(sigma_data, device='cuda')
    fid = torch_calculate_fid(mu, sigma, mu_data, sigma_data, atol=5e-4)
    intra_fid += float(fid.detach().cpu().numpy())
  # print(intra_fids, np.mean(intra_fids))
    
  return intra_fid / len(superclass_mapping.keys())
    
  
def super_class_mapping():
  class_to_superclass = [None] * 100
  for super_idx, class_indices in superclass_mapping.items():
    for class_idx in class_indices:
      class_to_superclass[class_idx] = super_idx
  return class_to_superclass