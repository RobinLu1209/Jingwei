import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)
from torch_geometric.data import Data
from torch_geometric.nn.dense.linear import Linear
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import scatter, mask_feature
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, ResGatedGraphConv, ChebConv, LEConv, GraphConv
from torch_geometric.nn import TopKPooling
from torch.nn import Tanh, Sigmoid
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, BatchNorm

from utils import *

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        if d_model % 2 == 1:
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, src, src, key_padding_mask=src_mask)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.linear2(F.relu(self.linear1(src)))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src

class PatternInterpolator(nn.Module):
    def __init__(self, pattern_bank, factor_dim, geo_dim, hidden_dim):
        super(PatternInterpolator, self).__init__()
        self.pattern_bank = pattern_bank
        self.bank_num, self.profile_len, self.pattern_dim = pattern_bank.shape
        self.factor_dim = 4
        self.geo_linear = nn.Linear(geo_dim, hidden_dim)
        self.geo_pattern = nn.Linear(self.profile_len*self.pattern_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(self.factor_dim, max_len=33)
        self.factor_attn = TransformerEncoderLayer(self.factor_dim, 2, hidden_dim)
        self.factor_pattern = nn.Linear(self.pattern_dim, self.factor_dim)
        self.score_linear = nn.Linear(self.bank_num, self.bank_num)
        self.norm = nn.LayerNorm(self.bank_num)
    
    def forward(self, x_geo, x_factor):
        geo_feature = self.geo_linear(x_geo)
        geo_pattern_input = torch.reshape(self.pattern_bank, (-1, self.profile_len*self.pattern_dim))
        geo_pattern = self.geo_pattern(geo_pattern_input)
        geo_score = torch.matmul(geo_feature, geo_pattern.T)
        x_factor = x_factor[:,:,:4]
        x_factor = self.positional_encoding(x_factor)
        factor_feature = self.factor_attn(x_factor)
        factor_pattern = self.factor_pattern(self.pattern_bank)
        factor_feature, factor_pattern = torch.reshape(factor_feature, (-1, self.profile_len*self.factor_dim)), torch.reshape(factor_pattern, (-1, self.profile_len*self.factor_dim))
        factor_score = torch.matmul(factor_feature, factor_pattern.T)

        score = self.norm(geo_score + factor_score)
        score = self.score_linear(score)

        return score, self.score_linear(self.norm(geo_score)), self.score_linear(self.norm(factor_score))


class OxygenGraphConv(MessagePassing):
    def __init__(self, input_dim, output_dim, edge_dim, meta_dim=0):
        super().__init__(aggr='add')
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_dim = edge_dim
        self.meta_dim = meta_dim
        self.norm = nn.LayerNorm(output_dim)
        self.build()

    def build(self):
        self.feature_transform = Linear(self.input_dim, self.output_dim, bias=False)
        self.edge_transform = Linear(self.edge_dim, 1)
        self.alpha_transform = nn.Linear(self.meta_dim, self.input_dim * self.input_dim)
        self.beta_transform = nn.Linear(self.meta_dim, self.input_dim)
    
    def forward(self, x, edge_index, edge_attr, meta_info):

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        alpha = self.alpha_transform(meta_info).view(-1, self.input_dim, self.input_dim)
        beta = self.beta_transform(meta_info)
        identity_matrix = torch.eye(self.input_dim).cuda()
        x = torch.bmm(alpha + identity_matrix, x.unsqueeze(2)).squeeze() + beta      
        x = self.feature_transform(x)

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm)
        out = out + x
        out = self.norm(out)
        return out, alpha, beta
    
    def message(self, x_j, edge_attr, norm):
        msg = norm.view(-1, 1) * x_j 
        return msg
    

class Jingwei(nn.Module):
    def __init__(self, pattern_bank, factor_dim, geo_dim, hidden_dim, time_len, layer_num, edge_dim, meta_dim):
        super(Jingwei, self).__init__()
        self.pattern_bank = pattern_bank
        self.bank_num, self.profile_len, self.pattern_dim = pattern_bank.shape
        self.layer_num = layer_num
        self.edge_dim, self.geo_dim, self.time_len, self.hidden_dim, self.meta_dim = edge_dim, geo_dim, time_len, hidden_dim, meta_dim
        self.pattern_interpolator = PatternInterpolator(pattern_bank, factor_dim, geo_dim, hidden_dim)
        self.pattern_output_linear = nn.Linear(self.pattern_dim, 1)
        self.do_encoder = TransformerEncoderLayer(d_model=self.profile_len, num_heads=1, d_ff=int(hidden_dim))
        self.positional_encoding = PositionalEncoding(self.profile_len, max_len=time_len)
        self.token_embedding = nn.Parameter(torch.zeros(self.profile_len, 1))
        self.depth_linear = nn.Linear(self.time_len, self.pattern_dim)
        self.geo_encoder = nn.Linear(geo_dim, hidden_dim)
        self.geo_decoder = nn.Linear(hidden_dim, self.profile_len)
        self.build_graph()
        self.output_linear = nn.Linear(self.hidden_dim, self.profile_len)
        self.spatial_decoding = nn.Linear(self.hidden_dim + self.profile_len * self.pattern_dim, 2)
    def build_graph(self):
        self.gnn_input_dim = self.profile_len * self.pattern_dim + 3 * self.profile_len
        gnn_list = [
            OxygenGraphConv(self.gnn_input_dim, self.hidden_dim, self.edge_dim, self.meta_dim)
        ]
        if self.layer_num > 1:
            for _ in range(self.layer_num - 1):
                gnn_list.append(OxygenGraphConv(self.hidden_dim, self.hidden_dim, self.edge_dim, self.meta_dim))
        self.gnn_layer = nn.ModuleList(gnn_list)

    def forward(self, x_geo, x_factor, do_series, edge_index, edge_attr, x_meta):
        alpha_ratio, geo_score, factor_score = self.pattern_interpolator(x_geo, x_factor)
        alpha_ratio_softmax_topk = process_scores(alpha_ratio, k_num=15)
        pattern_rep = torch.matmul(alpha_ratio_softmax_topk, self.pattern_bank.permute(1, 0, 2)).permute(1, 0, 2) 
        pattern_output = self.pattern_output_linear(pattern_rep).squeeze(-1)
        x_geo_embedding = self.geo_encoder(x_geo)
        x_geo_output = self.geo_decoder(torch.relu(x_geo_embedding))

        do_series[:, :, int(do_series.shape[-1]/2)] = self.token_embedding.squeeze()

        do_series = do_series.permute(0, 2, 1)
        do_series = self.positional_encoding(do_series)
        x_factor_main = x_factor[:,:,:3]
        do_encoder_input = do_series
        do_rep = self.do_encoder(do_encoder_input)
        profile_rep = do_rep
        profile_rep = self.depth_linear(profile_rep.permute(0, 2, 1))
        profile_rep = torch.reshape(profile_rep, (profile_rep.shape[0], -1))
        profile_rep = torch.cat((profile_rep, x_factor_main.reshape(x_factor_main.shape[0], -1)), dim=-1)
        gnn_input = torch.relu(profile_rep)
        for i in range(self.layer_num):
            gnn_output, alpha, beta = self.gnn_layer[i](gnn_input, edge_index, edge_attr, x_meta)
            if i != self.layer_num - 1:
                gnn_output = torch.relu(gnn_output)
            gnn_input = gnn_output
        time_series_output = self.output_linear(gnn_output)
        concat_feature = torch.cat((pattern_rep.reshape(-1, self.profile_len * self.pattern_dim), gnn_output), dim=-1)
        spatial_pred = self.spatial_decoding(concat_feature)
        oxygen_pred = torch.clamp(time_series_output + x_geo_output + pattern_output, min=0, max=500)
        return oxygen_pred, spatial_pred, time_series_output, x_geo_output, pattern_output
