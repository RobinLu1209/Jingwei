import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

def generate_random_mask(input_tensor, key_padding_mask):
    batch_size, seq_len, _ = input_tensor.size()
    random_mask = torch.zeros_like(key_padding_mask)
    random_mask_ratios = torch.rand(batch_size)

    for i in range(batch_size):
        random_mask_ratio = random_mask_ratios[i].item()
        valid_indices = (key_padding_mask[i] == 0).nonzero().squeeze(1)  
        num_masked = int(len(valid_indices) * random_mask_ratio)  
        while num_masked >= len(valid_indices):
            random_mask_ratio = torch.rand(1).item()
            num_masked = int(len(valid_indices) * random_mask_ratio)  
        masked_indices = valid_indices[torch.randperm(len(valid_indices))[:num_masked]]  
        random_mask[i, masked_indices] = 1  

    return random_mask.bool()

def norm_seq(sequences, masks, miu, sigma):
    seq_norm = (sequences - miu) / sigma
    zero_tensor = torch.zeros_like(seq_norm)
    output = torch.where(masks.unsqueeze(-1), zero_tensor, seq_norm)
    return output

def KMeans(tensor, k=20, max_iter=500):
    batch_size, profile_len, feature_dim = tensor.shape
    tensor = torch.reshape(tensor, (tensor.shape[0], -1)).numpy()
    centers = tensor[np.random.choice(tensor.shape[0], k, replace=False)]
    
    for _ in range(max_iter):
        print(_)
        distances = np.linalg.norm(tensor[:, np.newaxis] - centers, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centers = np.array([tensor[labels == i].mean(axis=0) for i in range(k)])
        if np.allclose(centers, new_centers):
            break
        centers = new_centers

    centers = torch.tensor(centers)
    centers = torch.reshape(centers, (-1, profile_len, feature_dim))
    return centers


def get_vaild_indices(data):
    data = (~torch.isnan(data)).sum(dim = 1)
    valid_indices = torch.nonzero(data)[:, 0]
    return valid_indices


def get_reg(torch_list):
    reg = 0
    for tensor in torch_list:
        if tensor is None:
            return 0
        reg += torch.norm(tensor)**2
    return reg

def get_softmax(input_list):
    exp_values = torch.exp((input_list - input_list.max())/(input_list.min() - input_list.max())) 
    return exp_values / exp_values.sum(axis=0)

def get_mock_weighted_loss(result, mock_result, label, metric='MSE'):

    if metric == 'MSE':
        criterion = nn.MSELoss(reduction='none')
    mock_loss_list = criterion(mock_result, label)
    true_loss_list = criterion(result, label)
    norm_score = get_softmax(mock_loss_list)
    weighted_sum = torch.dot(norm_score, true_loss_list)
    return weighted_sum


def process_scores(scores, k_num=8):
    top_indices = torch.topk(scores, k_num, dim=1).indices
    processed_scores = torch.zeros_like(scores)
    processed_scores.scatter_(1, top_indices, scores.gather(1, top_indices))
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(1, top_indices, True)
    softmax_scores = torch.zeros_like(scores)
    softmax_scores[mask] = F.softmax(processed_scores[mask].view(scores.size(0), -1), dim=1).view(-1)
    return softmax_scores
