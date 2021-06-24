import numpy as np
import cv2
from math import cos, sin
import torch
from pathlib import Path
import albumentations as albu
import glob
import h5py
import os

from torch.utils.data import Dataset

class HDF5_Dataset(Dataset):
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

        # CMU_dataset_64x64

        # self.dbs = self.load_dbs(base_dir, filename)
        self.dbs = self.load_all_hdf5(base_dir)
        # print(self.dbs[0]["labels"].shape[0])
        self.size_dbs = [db["labels"].shape[0] for db in self.dbs]
        self.numImages = sum(self.size_dbs)
        print(f"[INFO] Size dbs {self.numImages}")

    def load_dbs(self, dir, filename):
        file_paths = glob.glob(dir + "/*")
        file_names = [os.path.basename(file_path) for file_path in file_paths]
        # print(f"[INFO] File names :{file_names}")
        db_names = [file_name for file_name in file_names if file_name.startswith(filename)]
        db_names = sorted(db_names, key=lambda x: x.split("_")[-1], reverse=False)
        # print(f"[INFO] DB name: {db_names}")
        dbs = [h5py.File(os.path.join(dir, db_name)) for db_name in db_names]
        return dbs  

    def load_all_hdf5(self, dir):
        file_paths = glob.glob(dir + "/*")
        file_names = [os.path.basename(file_path) for file_path in file_paths]

        db_names = [file_name for file_name in file_names if file_name.endswith(".hdf5")]
        print(f"[INFO] Load training data from {db_names}")

        # db_names = ["/media/2tb/projects/VL's/UetHeadpose/pre_processed/val_data/UETHeadpose_dataset_64x64_0_3046.hdf5"]

        # print(f"[INFO] DB name: {db_names}")
        dbs = [h5py.File(os.path.join(dir, db_name)) for db_name in db_names]
        return dbs

    def get_true_index(self, index):
        for i, size_db in enumerate(self.size_dbs):
            if index < size_db:
                return index, i
            else:
                index -= size_db
                
    def __len__(self):
        return self.numImages

    def get_one_img(self, index):

        index, num_db = self.get_true_index(index)

        image = self.dbs[num_db]["images"][index]
        label = self.dbs[num_db]["labels"][index]

        # print(label)
        # print(f"[INFO] Label: {label}")

        # ImageAugment (RandomBrightness, AddNoise...)
        if self.image_augmenter:
            augmented = self.image_augmenter(image=image)
            image = augmented['image']

        # Resize (Scale & Pad & Crop)
        # if self.resizer:
        #     resized = self.resizer(image=image)
        #     image = resized['image']
        # AffineAugment (Horizontal Flip, Rotate...)
        if self.affine_augmenter:
            augmented = self.affine_augmenter(image=image)
            image = augmented['image']

        if self.debug:
            # print(label)
            return image, label
        else:
            
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

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 80, radian=False):
    if not radian:
       
        pitch = pitch * np.pi / 180
        yaw = -(yaw * np.pi / 180)
        roll = roll * np.pi / 180
        # print(pitch)

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import random
    from PIL import Image

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
    
    dataset = HDF5_Dataset(base_dir="/media/2tb/projects/VL's/UetHeadpose/pre_processed", affine_augmenter=None, image_augmenter=None,
                             filename='UET', target_size=64, debug=True)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=1, pin_memory=True, drop_last=True)
    print(len(dataset))

    dir_path = Path("./train_img")
    dir_path.mkdir(parents=True, exist_ok=True)

    # for i, batched in enumerate(dataloader):
    #     images, labels, _ , _, _ = batched
    #     for j in range(8):
    #         img = img1[j].numpy()
    #         img = img.astype('uint8')
    #         img = Image.fromarray(img)
    #         img.save(dir_path / f'300W_img1_{i}_{j}.jpg')
    #         img = img2[j].numpy()
    #         img = img.astype('uint8')
    #         img = Image.fromarray(img)
    #         # img.save(dir_path / '300W_img2_{i}_{j}.jpg')
    #     if i > 2:
    #         break

    # with tqdm(dataloader) as _tqdm:
    #     for batched in _tqdm:
    #         print(".")


    for i in range(0, 3000, 100):    
        img, label = dataset.get_one_img(i)
        img = np.array(img)
        print(img.shape)
        print(img.dtype)
        img = draw_axis(img, label[0], label[1], label[2])
        img = Image.fromarray(img)
        img.save(dir_path / f'Uet_val_img{i}.jpg')