import numpy as np
import torch
import random
import argparse
from pathlib import Path
import yaml
import albumentations as albu
from torch.utils.data import DataLoader
from collections import OrderedDict
import torch.nn as nn
from tqdm import tqdm
from dataset import load_dataset

from models import load_model
from losses import Criterion
from logger.plot import history_ploter

from utils.optimizer import create_optimizer
from utils.metrics import calculate_diff


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
    train_config = config['Train']

    

    # Config for data:
    train_dir = data_config["train_dir"]
    train_name = data_config["train_name"]
    train_type = data_config["train_type"]

    val_dir = data_config["val_dir"]
    val_name = data_config["val_name"]
    val_type = data_config["val_type"]

    target_size = data_config["target_size"]
    num_workers = data_config["num_worker"]

    # Config for train:
    num_epoch = train_config["num_epoch"]
    batch_size = train_config["batch_size"]
    val_every = train_config["val_every"]
    resume = train_config["val_every"]
    pretrained_path = train_config["pretrained_path"]
    loss_type = train_config["loss_type"]
    optimizer_config = train_config["optimizer"]

    del data_config
    del train_config

    model = load_model(**net_config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # To device
    model = model.to(device)
    # if torch.cuda.is_available():
    #     model.cuda()

    modelname = config_path.stem
    output_dir = Path('../model_train') / modelname
    output_dir.mkdir(parents=True ,exist_ok=True)
    log_dir = Path('../logs') / modelname
    log_dir.mkdir(parents=True , exist_ok=True)

    # logger = debug_logger(log_dir)
    # logger.debug(config)
    # logger.info(f'Device: {device}')
    # logger.info(f'Max Epoch: {max_epoch}')
    
    loss_fn = Criterion(loss_type=loss_type).to(device)
    params = model.parameters()
    optimizer, scheduler = create_optimizer(params, **optimizer_config)

    # Dataset
    affine_augmenter = albu.Compose([albu.GaussNoise(var_limit=(0,25),p=.2),
                                    albu.GaussianBlur(3, p=0.2),
                                    albu.JpegCompression(50, 100, p=0.2)])

    image_augmenter = albu.Compose([
                                    albu.OneOf([
                                        albu.RandomBrightnessContrast(0.25,0.25),
                                        albu.CLAHE(clip_limit=2),
                                        albu.RandomGamma(),
                                        ], p=0.5),
                                    albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20,p=0.2),
                                    albu.RGBShift(p=0.2),
                                    ])
    train_dataset = load_dataset(data_type=train_type, base_dir=train_dir, filename=train_name, n_class=net_config['n_class'], target_size=target_size, 
        affine_augmenter=affine_augmenter, image_augmenter=image_augmenter, debug=False )
    val_dataset = load_dataset(data_type=val_type, base_dir=val_dir, filename=val_name, n_class=net_config['n_class'], target_size=target_size, debug=False )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    if torch.cuda.is_available():
        model = nn.DataParallel(model)

    # model = model.load_state_dict(torch.load("/home/linhnv/projects/fsanet_pytorch/model_train/headpose_resnet/model_epoch_77_6.682497448080261.pth"))    
    # optimizer = optimizer.load_state_dict(torch.load("/home/linhnv/projects/fsanet_pytorch/model_train/headpose_resnet/opt_epoch_77_6.682497448080261.pth"))

    # Restore model
    # torch.save({
    #             'epoch': i_epoch,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'loss_history': loss_history,
    #             }, output_dir.joinpath(f'checkpoint_epoch_{i_epoch}_{valid_diff}.pth'))

    if False:

        print("[INFO] resume training.")
        # model_path = output_dir.joinpath(f'model_epoch_{start_epoch-1}.pth')
        # logger.info(f'Resume from {model_path}')
        model_path = "/home/linhnv/projects/fsanet_pytorch/model_train/headpose_resnet/model_epoch_67_5.604301367075212.pth"
        param = torch.load(model_path, map_location='cpu')
        model.load_state_dict(param)
        model.to(device)
        del param
        # opt_path = output_dir.joinpath(f'opt_epoch_{start_epoch-1}.pth')
        opt_path = "/home/linhnv/projects/fsanet_pytorch/model_train/headpose_resnet/opt_epoch_67_5.604301367075212.pth"
        param = torch.load(opt_path)
        optimizer.load_state_dict(param)
        del param

    best_metrics = float('inf')
    loss_history = []
    # Train
    for i_epoch in range(0, num_epoch):
        print(f"Epoch: {i_epoch}")
        train_losses = []
        train_diffs = []
        model.train()
        with tqdm(train_loader) as _tqdm:
            for batched in _tqdm:
                optimizer.zero_grad()

                # if loss_type == "RANK":
                
                img1, img2, lbl1, lbl2, labels = batched
                img1, img2, lbl1, lbl2, labels = img1.to(device),img2.to(device),lbl1.to(device),lbl2.to(device),labels.to(device)

                preds1 = model(img1)
                preds2 = model(img2)
                
                loss = loss_fn([preds1,preds2], [lbl1,lbl2,labels])

                diff = calculate_diff(preds1, lbl1)
                diff += calculate_diff(preds2, lbl2)
                diff /= 2

                _tqdm.set_postfix(OrderedDict(loss=f'{loss.item():.3f}', mae=f'{diff:.1f}'))
                train_losses.append(loss.item())
                train_diffs.append(diff)

                loss.backward()
                optimizer.step()

        train_loss = np.mean(train_losses)
        train_diff = np.nanmean(train_diffs)
        
        print(f'[INFO] train loss: {train_loss}')
        print(f'[INFO] train diff: {train_diff}')

        scheduler.step()

        if (i_epoch + 1) % val_every == 0:
            valid_losses = []
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
                        valid_losses.append(0)
                        valid_diffs.append(diff)

            valid_loss = np.mean(valid_losses)
            valid_diff = np.mean(valid_diffs)
            loss_history.append([train_loss, valid_loss])
            history_ploter(loss_history, log_dir.joinpath('loss.png'))

            print(f'[INFO] valid seg loss: {valid_loss}')
            print(f'[INFO] valid diff: {valid_diff}')

            best_metrics = valid_diff
            print('Best Model!\n')
            torch.save(model.state_dict(), output_dir.joinpath(f'model_epoch_{i_epoch}_{valid_diff}.pth'))
            torch.save({
                'epoch': i_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_history': loss_history,
                }, output_dir.joinpath(f'checkpoint_epoch_{i_epoch}_{valid_diff}.pth'))

        else:
            valid_loss = None
            valid_diff = None

if __name__ == "__main__":
    main()
