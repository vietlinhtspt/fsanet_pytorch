# -*- coding: utf-8
import yaml
import torch
import pickle
import random
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from losses import Criterion
from models import load_model
import albumentations as albu
from dataset import load_dataset
from collections import OrderedDict
from logger.plot import history_ploter
from torch.utils.data import DataLoader
from utils.metrics import calculate_diff
from glob import glob

seed = 2020
np.random.seed(seed)
random.seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('config_path')
args = parser.parse_args()

def main():
    config_path = Path(args.config_path)
    config = yaml.load(open(config_path))

    net_config = config['Net']
    data_config = config['Data']
    evaluate_config = config['Evaluate']

    val_dir = evaluate_config['eval_dir']
    val_name = evaluate_config['eval_name']
    val_type = evaluate_config['eval_type']

    target_size = data_config["target_size"]
    use_bined = False
    num_workers = data_config['num_worker']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # pretrained_paths = ["/media/2tb/output_models/headpose_resnet/model_epoch_79_4.179329837522199.pth"] # 4.6
    pretrained_paths = ["/media/2tb/output_models/headpose_resnet/model_epoch_78_3.985460854345752.pth"] # 4.3
    # pretrained_paths = ["/media/2tb/output_models/headpose_fsanet/model_epoch_228_5.439456623458148.pth"] # 7.8


    # models_path = glob("/home/linhnv/projects/RankPose/model/headpose_resnet/*")
    # models_path = [x for x in models_path if x.startswith("/home/linhnv/projects/RankPose/model/headpose_resnet/model_epoch")]
    # print(models_path)

    model = load_model(**net_config)
    # To device
    model = model.to(device)

    modelname = config_path.stem
    output_dir = Path('../model') / modelname
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path('../logs') / modelname
    log_dir.mkdir(parents=True, exist_ok=True)

    # logger = debug_logger(log_dir)
    # logger.debug(config)
    # logger.info(f'Device: {device}')

    valid_dataset = load_dataset(data_type=val_type, base_dir=val_dir, filename=val_name, n_class=net_config['n_class'], target_size=target_size, debug=False)

    # top_10 = len(train_dataset) // 10
    # top_30 = len(train_dataset) // 3.33
    # train_weights = [ 3 if idx<top_10 else 2 if idx<top_30 else 1 for idx in train_dataset.labels_sort_idx]
    # train_sample = WeightedRandomSampler(train_weights, num_samples=len(train_dataset), replacement=True)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sample, num_workers=num_workers,
    #                           pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True)

    if torch.cuda.is_available():
        model = nn.DataParallel(model)

    

    for pretrained_path in pretrained_paths:
        print(f"[INFO] Pretrained path: {pretrained_path}")

        # logger.info(f'Load pretrained from {pretrained_path}')
        param = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(param)
        model.to(device)
        del param
        
        valid_diffs = []
        model.eval()
        with torch.no_grad():
            with tqdm(valid_loader) as _tqdm:
                for batched in _tqdm:
                    
                    images, labels = batched
                
                    images, labels = images.to(device), labels.to(device)

                    preds = model(images)
                
                    # loss = loss_fn([preds], [labels])

                    diff = calculate_diff(preds, labels)
                    
                    _tqdm.set_postfix(OrderedDict(mae=f'{diff:.2f}'))
                    # _tqdm.set_postfix(OrderedDict(loss=f'{loss.item():.3f}', d_y=f'{np.mean(diff[:,0]):.1f}', d_p=f'{np.mean(diff[:,1]):.1f}', d_r=f'{np.mean(diff[:,2]):.1f}'))
                  
                    valid_diffs.append(diff)

       
        valid_diff = np.mean(valid_diffs)
        print(f'valid diff: {valid_diff}')
        # logger.info(f'valid seg loss: {valid_loss}')
        # logger.info(f'valid diff: {valid_diff}')

if __name__=='__main__':
    main()