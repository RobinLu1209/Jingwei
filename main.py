import argparse
import gzip
import torch
import pdb
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader
from torch.optim.lr_scheduler import LambdaLR, StepLR
from torch_geometric.loader import NeighborLoader
from models import *
from utils import *
import os
import random
from tqdm import tqdm

import pandas as pd

def get_vaild_indices(data, year, spilt_indices):
    start_year = 1920
    data = (~torch.isnan(data)).sum(dim = 1)
    valid_indices = torch.nonzero(data)[:, 0]
    spilt_indices = spilt_indices[year - start_year]
    valid_indices = torch.tensor(list(set(valid_indices.numpy()) & set(spilt_indices)))
    return valid_indices

def custom_shuffle(rank_arr, select_arr):
    other_elements = np.setdiff1d(rank_arr, select_arr)
    np.random.shuffle(other_elements)
    result_arr = np.concatenate((select_arr, other_elements))
    return result_arr

def transform_depth(depth_arr, depth_range):
    transformed_arr = np.zeros_like(depth_arr, dtype=int)
    for i, value in enumerate(depth_arr):
        for j in range(len(depth_range)):
            if j < len(depth_range) - 1:
                if depth_range[j] <= value < depth_range[j + 1]:
                    transformed_arr[i] = j
                    break
            else:
                if value >= depth_range[j]:
                    transformed_arr[i] = j
                    break
    return transformed_arr

def xy_normalization(longitude_i, latitude_i):
    longitude_i = np.where(longitude_i > 180, longitude_i-360, longitude_i)
    longitude_i = np.where(longitude_i >= 0, np.floor(longitude_i) + 0.5, np.ceil(longitude_i) - 0.5)
    latitude_i = np.where(latitude_i >= 0, np.floor(latitude_i) + 0.5, np.ceil(latitude_i) - 0.5)
    combined_arr = np.column_stack((longitude_i, latitude_i))
    unique_combined_arr = np.unique(combined_arr, axis=0)
    unique_x_arr = unique_combined_arr[:, 0]
    unique_y_arr = unique_combined_arr[:, 1]
    return unique_x_arr, unique_y_arr

def spilt_dataset(num_graphs=104, num_nodes_per_graph=42491, seed=42):
    indices_matrix = np.tile(np.arange(num_nodes_per_graph), (num_graphs, 1))
    np.random.seed(seed)
    for i in range(indices_matrix.shape[0]):
        np.random.shuffle(indices_matrix[i])
    train_ratio = 0.7 
    val_ratio = 0.1 
    test_ratio = 0.2 
    num_train = int(train_ratio * num_nodes_per_graph)
    num_val = int(val_ratio * num_nodes_per_graph)
    train_indices = indices_matrix[:, :num_train]
    val_indices = indices_matrix[:, num_train:num_train + num_val]
    test_indices = indices_matrix[:, num_train + num_val:]
    return train_indices, val_indices, test_indices

def get_args():
    parser = argparse.ArgumentParser(description='Transformer Model Arguments')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in the Transformer model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--max_patience', type=int, default=10, help='Maximum patience for early stopping')
    parser.add_argument('--lr', type=float, default=1e-1, help='Learning rate for training')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs for training')
    parser.add_argument('--gpu', type=int, default=0, help='Index of GPU to use for training')
    parser.add_argument('--geo_dim', type=int, default=5, help='The dimension of the DO geo factor')
    parser.add_argument('--hidden_dim', type=int, default=64, help='The dimension of the hidden layer')
    parser.add_argument('--time_length', type=int, default=11, help='The length of the time series')
    parser.add_argument('--edge_dim', type=int, default=3, help='The number of edge features')
    parser.add_argument('--input_dim', type=int, default=10, help='The number of edge features')
    parser.add_argument('-r', '--with_spatial_reg', action='store_true')
    parser.add_argument('--time_token', action='store_true')
    parser.add_argument('--tips', type=str, default='None', help='tips')
    parser.add_argument('--pattern_bank', type=str, default='svd_all', help='tips')
    parser.add_argument('--model', type=str, default='tree', help='tips')
    parser.add_argument('--seed', type=int, default=None, help='seed')
    parser.add_argument('--start_epoch', type=int, default=0, help='start_epoch')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    if args.seed is None:
        args.seed = random.randint(0, 10000)
    print(f"INFO: Using seed {args.seed}")
    print(args)
    pwd = "your_dataset"
    num_years = 104
    num_nodes = 42491
    train_indices, val_indices, test_indices = spilt_dataset(num_graphs=num_years, num_nodes_per_graph=num_nodes, seed=args.seed)
    pattern_bank = torch.load('data/pattern_vector.pt').to(device)
    
    model = Jingwei(pattern_bank, factor_dim=args.input_dim, geo_dim=args.geo_dim, hidden_dim=args.hidden_dim, time_len=args.time_length, layer_num=args.num_layers, edge_dim=args.edge_dim, meta_dim=args.geo_dim).to(device)
    criterion = nn.MSELoss()
    var_criterion = nn.MSELoss(reduction='none')
    if args.pattern_bank == 'param':
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': pattern_bank}], lr=args.lr, weight_decay=1e-5)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    decay_factor = 0.95
    initial_lr = args.lr
    min_lr = 1e-4
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: decay_factor ** epoch if initial_lr * (decay_factor ** epoch) >= min_lr else min_lr / initial_lr)
    patience = 0
    min_val_loss = 50000000
    break_outer = False

    if args.start_epoch!=0:
        model.load_state_dict(torch.load(f'result/model_{args.hidden_dim}_{args.tips}_{args.seed}.pt'))
        args.lr = 0.0001
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': pattern_bank}], lr=args.lr, weight_decay=1e-5)
    for epoch in range(args.start_epoch, args.num_epochs):
        model.train()
        iters = 0
        year_list = [num for num in range(1920, 2024)]
        random.shuffle(year_list)
        for year in year_list[0:30]:
            file_path = pwd + str(year) + '.pt'
            data = torch.load(file_path)
            indices = get_vaild_indices(data.y, year, train_indices)
            train_loader = NeighborLoader(data, num_neighbors=[-1, -1], batch_size=args.batch_size, input_nodes=indices)
            for i, batch in enumerate(train_loader):
                batch = batch.to(device)
                result, spatial_pred, time_series_output, x_geo_output, pattern_output = model(batch.x_geo.float(), batch.x, batch.time_series_profile, batch.edge_index, batch.edge_attr, batch.x_geo.float())
                xyz = batch.x_geo[:,:3]
                latitude = torch.arcsin(xyz[:,-1])
                latitude = torch.sin(torch.rad2deg(latitude))
                longitude = torch.arctan2(xyz[:,1],xyz[:,0])
                longitude = torch.rad2deg(longitude)
                longitude = torch.sin(longitude/360*3.1415926)
                spatial_label = torch.cat((latitude.unsqueeze(1), longitude.unsqueeze(1)), dim=1)
                year_label = torch.full((latitude.shape[0], 1), year).cuda()
                ssl_label = spatial_label
                spatial_loss = criterion(spatial_pred, ssl_label)
                y = batch.y[:len(batch.input_id)]
                mask = ~torch.isnan(y)
                mseLoss_list = var_criterion(result[:len(batch.input_id)][mask], y[mask]) 
                mean_mseloss, std_mseloss = torch.mean(mseLoss_list), torch.std(mseLoss_list)
                loss = mean_mseloss + 0.1 * spatial_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iters += 1
                if iters % 10 == 0:
                    print(f'Epoch {epoch}, Year_ID {year}, Mean MSE Loss: {round(mean_mseloss.item(),3)}, Spatial Loss:{round(spatial_loss.item(), 3)} output_max:{round(result[:len(batch.input_id)][mask].max().item(),3)}, output_min:{round(result[:len(batch.input_id)][mask].min().item(),3)}, lr: {scheduler.get_last_lr()}')
                    print(f'Average x_geo {round(x_geo_output.mean().item(), 2)}, average time_series {round(time_series_output.mean().item(), 2)}, average pattern {round(pattern_output.mean().item(), 2)}')
        scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            year_list = [num for num in range(1920, 2024)]
            count = 0
            for year in tqdm(year_list):
                file_path = pwd + str(year) + '.pt'
                data = torch.load(file_path)
                indices = get_vaild_indices(data.y, year, val_indices)
                val_loader = NeighborLoader(data, num_neighbors=[-1, -1], batch_size=args.batch_size, input_nodes=indices)
                for i, batch in enumerate(val_loader):
                    batch = batch.to(device)
                    result, _, _, _, _ = model(batch.x_geo.float(), batch.x, batch.time_series_profile, batch.edge_index, batch.edge_attr, batch.x_geo.float())
                    pred = result[:len(batch.input_id)]
                    y = batch.y[:len(batch.input_id)]
                    mask = ~torch.isnan(y)
                    loss = criterion(pred[mask], y[mask])
                    val_loss += loss.item()
                    count += 1

            val_loss = val_loss / count
            print(f'Epoch {epoch}, Validation Loss: {val_loss}, tips: {args.tips}')
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                patience = 0
                torch.save(model.state_dict(), f'result/model_{args.hidden_dim}_{args.tips}_{args.seed}.pt')
            else:
                patience += 1
                if patience >= args.max_patience:
                    break_outer = True
                    break
            
    # Test
    model.load_state_dict(torch.load(f'result/model_{args.hidden_dim}_{args.tips}_{args.seed}.pt'))
    model.eval()
    test_loss = 0
    with torch.no_grad():
        year_list = [num for num in range(1920, 2024)]
        count = 0
        for year in tqdm(year_list):
            file_path = pwd + str(year) + '.pt'
            data = torch.load(file_path)
            indices = get_vaild_indices(data.y, year, test_indices)
            test_loader = NeighborLoader(data, num_neighbors=[-1, -1], batch_size=args.batch_size, input_nodes=indices)
            for i, batch in enumerate(test_loader):
                batch = batch.to(device)
                result, _, _, _, _, _, _ = model(batch.x_geo.float(), batch.x, batch.time_series_profile, batch.edge_index, batch.edge_attr, batch.x_geo.float())
                pred = result[:len(batch.input_id)]
                y = batch.y[:len(batch.input_id)]
                mask = ~torch.isnan(y)
                loss = criterion(pred[mask], y[mask])
                test_loss += loss.item()
                count +=1
        test_loss = test_loss / count
        print(f'Test Loss: {test_loss}')


if __name__ == "__main__":
    main()