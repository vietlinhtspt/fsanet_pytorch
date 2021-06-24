import yaml
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import albumentations as albu
from collections import OrderedDict
from torch.utils.data import DataLoader


from dataset import load_dataset
from models import load_model
from losses import Criterion
from logger.plot import history_ploter
from utils.optimizer import create_optimizer
from utils.metrics import calculate_diff, calculate_diff_MAWE

seed = 2020
np.random.seed(seed)
random.seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# parser = argparse.ArgumentParser()
# parser.add_argument('config_path')
# args = parser.parse_args()

def main(pretrained_path, config_path, calculate_diff, val_dir, val_name, val_type):
    """
    input:
        pretrained_path: path to checkpoints
        config_path: path to config
        calculate_diff: Metric function
        val_dir, val_name, val_type: config for data
    output:
        None
    """
    # config_path = Path(args.config_path)
    config = yaml.load(open(config_path))

    net_config = config['Net']
    data_config = config['Data']
    train_config = config['Train']

    target_size = data_config["target_size"]
    num_workers = data_config["num_worker"]

    batch_size = train_config["batch_size"]
    
    del data_config
    del train_config

    model = load_model(**net_config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")
    # To device
    model = model.to(device)
    # if torch.cuda.is_available():
    #     model.cuda()   

    # logger = debug_logger(log_dir)
    # logger.debug(config)
    # logger.info(f'Device: {device}')
    # logger.info(f'Max Epoch: {max_epoch}')

    val_dataset = load_dataset(data_type=val_type, base_dir=val_dir, filename=val_name, n_class=net_config['n_class'], target_size=target_size, debug=False )

    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    if torch.cuda.is_available():
        model = nn.DataParallel(model)


    checkpoint = torch.load(pretrained_path)
    model.load_state_dict(checkpoint['model_state_dict'])    
        
        
    valid_diffs = []
    model.eval()
    with torch.no_grad():
        with tqdm(valid_loader) as _tqdm:
            for batched in _tqdm:
                
                # images, labels = batched
                images, labels, _ , _, _ = batched
            
                images, labels = images.to(device), labels.to(device)

                preds = model(images)
            
                # loss = loss_fn([preds], [labels])

                dif_yaw, dif_pitch, dif_roll, diff = calculate_diff(preds, labels, True)
                
                # print(diff, dif_yaw, dif_pitch, dif_roll)
                
                _tqdm.set_postfix(OrderedDict(mae=f'{diff:.2f}'))
                # _tqdm.set_postfix(OrderedDict(loss=f'{loss.item():.3f}', d_y=f'{np.mean(diff[:,0]):.1f}', d_p=f'{np.mean(diff[:,1]):.1f}', d_r=f'{np.mean(diff[:,2]):.1f}'))
                
                valid_diffs.append([dif_yaw, dif_pitch, dif_roll])

    valid_diffs = np.array(valid_diffs)
    print(valid_diffs.shape)
    valid_diff = np.mean(valid_diffs, 0)
    
    print(f'[INFO] valid diff: {valid_diff}')
    print((valid_diff[0] + valid_diff[1] + valid_diff[2]) / 3) 

def create_model(config_path):
    """
    input:
        config_path: Path to config
    output:
        model: model
        target_size, num_workers, batch_size, net_config['n_class']: Parameters in config file.
    """
    config = yaml.load(open(config_path))

    net_config = config['Net']
    data_config = config['Data']
    train_config = config['Train']

    target_size = data_config["target_size"]
    num_workers = data_config["num_worker"]

    batch_size = train_config["batch_size"]
    
    del data_config
    del train_config

    model = load_model(**net_config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")
    # To device
    model = model.to(device)

    if torch.cuda.is_available():
        model = nn.DataParallel(model)

    return model, target_size, num_workers, batch_size, net_config['n_class']

def load_pretrain_model(model, pretrained_path):
    checkpoint = torch.load(pretrained_path)
    model.load_state_dict(checkpoint['model_state_dict'])    
    return model

def evaluate_fusion(pretrained_uni_path, config_uni_path, pretrained_var_path, config_var_path, pretrained_wei_path, config_wei_path, calculate_diff, val_dir, val_name, val_type):
    """
    input:
        pretrained_uni_path, pretrained_var_path, pretrained_wei_path: pretrained paths of 3 model types.
        config_uni_path, config_var_path, config_wei_path: config paths of 3 model types.
        calculate_diff: Metric function
        val_dir, val_name, val_type: config for data
    output:
        None
    """

    # config_path = Path(args.config_path)
    uni_model, target_size, num_workers, batch_size, n_class = create_model(config_uni_path)
    var_model, target_size, num_workers, batch_size, n_class = create_model(config_var_path)
    wei_model, target_size, num_workers, batch_size, n_class = create_model(config_wei_path)

    val_dataset = load_dataset(data_type=val_type, base_dir=val_dir, filename=val_name, n_class=n_class, target_size=target_size, debug=False )

    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    uni_model = load_pretrain_model(uni_model, pretrained_uni_path)
    var_model = load_pretrain_model(var_model, pretrained_var_path)
    wei_model = load_pretrain_model(wei_model, pretrained_wei_path)

    valid_diffs = []

    uni_model.eval()
    var_model.eval()
    wei_model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        with tqdm(valid_loader) as _tqdm:
            for batched in _tqdm:
                
                # images, labels = batched
                images, labels, _ , _, _ = batched
            
                images, labels = images.to(device), labels.to(device)

                uni_preds = uni_model(images)
                var_preds = var_model(images)
                wei_preds = wei_model(images)

                preds = uni_preds.add(var_preds).add(wei_preds)/3
            
                # loss = loss_fn([preds], [labels])

                dif_yaw, dif_pitch, dif_roll, diff = calculate_diff(preds, labels, True)
                # print(diff, dif_yaw, dif_pitch, dif_roll)
                
                _tqdm.set_postfix(OrderedDict(mae=f'{diff:.2f}'))
                # _tqdm.set_postfix(OrderedDict(loss=f'{loss.item():.3f}', d_y=f'{np.mean(diff[:,0]):.1f}', d_p=f'{np.mean(diff[:,1]):.1f}', d_r=f'{np.mean(diff[:,2]):.1f}'))
                
                valid_diffs.append([dif_yaw, dif_pitch, dif_roll, diff])
    valid_diffs = np.array(valid_diffs)
    print(valid_diffs.shape)
    valid_diff = np.mean(valid_diffs, 0)
    
    print(f'[INFO] valid diff: {valid_diff}')     
    print((valid_diff[0] + valid_diff[1] + valid_diff[2]) / 3) 

if __name__ == "__main__":

    # wei_model_path = "/media/2tb/projects/VL's/FSANet/models/fsanet_origial_wei_MSE_CMU/checkpoint_epoch_147_17.134625017642975.pth"
    
    # 1x1
    # uni_model_path = "/home/linhnv/projects/fsanet_pytorch/model/mse_300WLP/uni_model.pth"
    # uni_model_path = "/home/linhnv/projects/fsanet_pytorch/model/wrapped_300WLP/uni_model.pth"
    # uni_model_path = "/home/linhnv/projects/fsanet_pytorch/model/mse_cmu/uni_model.pth"
    # uni_model_path = "/home/linhnv/projects/fsanet_pytorch/model/mse_uet/uni_model.pth"
    uni_model_path = "/home/linhnv/projects/fsanet_pytorch/model/mse_uet_113/uni_model.pth"

    # var
    # var_model_path = "/home/linhnv/projects/fsanet_pytorch/model/mse_300WLP/var_model.pth"
    # var_model_path = "/home/linhnv/projects/fsanet_pytorch/model/wrapped_300WLP/var_model.pth"
    # var_model_path = "/home/linhnv/projects/fsanet_pytorch/model/mse_cmu/var_model.pth"
    # var_model_path = "/home/linhnv/projects/fsanet_pytorch/model/mse_uet/var_model.pth"
    var_model_path = "/home/linhnv/projects/fsanet_pytorch/model/mse_uet_113/var_model.pth"

    # w/o
    # wei_model_path = "/home/linhnv/projects/fsanet_pytorch/model/mse_300WLP/wei_model.pth" 
    # wei_model_path = /home/linhnv/projects/fsanet_pytorch/model/wrapped_300WLP/wei_model.pth
    # wei_model_path = "/home/linhnv/projects/fsanet_pytorch/model/mse_cmu/wei_model.pth"
    # wei_model_path = "/home/linhnv/projects/fsanet_pytorch/model/mse_uet/wei_model.pth" 
    wei_model_path = "/home/linhnv/projects/fsanet_pytorch/model/mse_uet_113/wei_model.pth"

    
    uni_config_path = "/home/linhnv/projects/fsanet_pytorch/config/fsanet_uni_validate.yaml"
    var_config_path = "/home/linhnv/projects/fsanet_pytorch/config/fsanet_var_validate.yaml"
    wei_config_path = "/home/linhnv/projects/fsanet_pytorch/config/fsanet_wei_validate.yaml"

    val_dir = "/media/2tb/projects/VL's/UetHeadpose/pre_processed/val_data_3"
    val_name = "CMU_dataset_64x64"
    val_type = "HDF5_multi"

    val_dir = "/home/linhnv/projects/RankPose/data"
    val_name = "aflw2000_filename.txt"
    val_type = "AFLW2000"

    evaluate_fusion(uni_model_path, uni_config_path, var_model_path, var_config_path, wei_model_path, wei_config_path, calculate_diff_MAWE, val_dir, val_name, val_type)
    
    main(uni_model_path, uni_config_path, calculate_diff_MAWE, val_dir, val_name, val_type)
    main(var_model_path, var_config_path, calculate_diff_MAWE, val_dir, val_name, val_type)
    main(wei_model_path, wei_config_path, calculate_diff_MAWE, val_dir, val_name, val_type)