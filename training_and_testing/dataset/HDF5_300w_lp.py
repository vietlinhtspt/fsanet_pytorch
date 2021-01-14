
import os
import torch
import numpy as np
import tqdm
from PIL import Image
from pathlib import Path
import albumentations as albu
import pickle
import h5py
from torch.utils.data import Dataset

import sys
sys.path.append("..")

from utils.preprocess import preprocess, change_bbox
from utils.functional import get_pt_ypr_from_mat

class Rank300WLP_HDF5_Dataset(Dataset):
    def __init__(self, base_dir=None, filename=None, n_class=3, target_size=224, 
        affine_augmenter=None, image_augmenter=None, debug=False, paired_img=False):

        print("[INFO] Initing Rank300WLPDataset with HDF5 type.")
        print(base_dir, filename, n_class, target_size, debug)
        self.base_dir = Path(base_dir)
        self.n_class = n_class
        self.target_size = target_size
        self.affine_augmenter = affine_augmenter
        self.image_augmenter = image_augmenter
        self.debug = debug
        self.paired_img = paired_img
        self.CMU_data = False

        self.resizer = albu.Compose([albu.SmallestMaxSize(target_size, p=1.),
                                        albu.CenterCrop(target_size, target_size, p=1.)])

        self.db = h5py.File(os.path.join(base_dir, filename))
        self.numImages = self.db["labels"].shape[0]
                
    def __len__(self):
        return self.numImages

    def get_one_img(self, index):

        image = self.db["images"][index]
        label = self.db["labels"][index]

        # print(label)
        # print(f"[INFO] Label: {label}")

        # ImageAugment (RandomBrightness, AddNoise...)
        if self.image_augmenter:
            augmented = self.image_augmenter(image=image)
            image = augmented['image']

        # Resize (Scale & Pad & Crop)
        if self.resizer:
            resized = self.resizer(image=image)
            image = resized['image']
        # AffineAugment (Horizontal Flip, Rotate...)
        if self.affine_augmenter:
            augmented = self.affine_augmenter(image=image)
            image = augmented['image']

        if self.debug:
            # print(label)
            return image
        else:
            image = preprocess(image)
            image = torch.FloatTensor(image).permute(2, 0, 1)
            # print(label)
            label = torch.FloatTensor(label)

            # print(f"[INFO] Label: {label.shape}")
            
            return image, label, image, label, label

    def __getitem__(self, index):
        if self.paired_img:
            return self.get_two_imgs(index)
        else:
            return self.get_one_img(index)


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import random

    seed = 2020
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
    dataset = Rank300WLP_HDF5_Dataset(base_dir="/home/linhnv/projects/RankPose/data", affine_augmenter=affine_augmenter, image_augmenter=image_augmenter,
                             filename='300w_lp_for_rank.txt', target_size=224, debug=False)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True, drop_last=True)
    print(len(dataset))

    dir_path = Path("./train_img")
    dir_path.mkdir(parents=True, exist_ok=True)

    # for i, batched in enumerate(dataloader):
    #     img1, img2,  lbl1, lbl2, label = batched
    #     for j in range(8):
    #         img = img1[j].numpy()
    #         img = img.astype('uint8')
    #         img = Image.fromarray(img)
    #         # img.save(dir_path / f'300W_img1_{i}_{j}.jpg')
    #         img = img2[j].numpy()
    #         img = img.astype('uint8')
    #         img = Image.fromarray(img)
    #         # img.save(dir_path / '300W_img2_{i}_{j}.jpg')
    #     if i > 2:
    #         break

    with tqdm(dataloader) as _tqdm:
        for batched in _tqdm:
            print(".")


