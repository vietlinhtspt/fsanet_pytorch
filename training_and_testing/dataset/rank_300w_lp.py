
import os
import torch
import numpy as np
import tqdm
from PIL import Image
from pathlib import Path
import albumentations as albu
from torch.utils.data import Dataset

import sys
sys.path.append("..")

from utils.preprocess import preprocess, change_bbox
from utils.functional import get_pt_ypr_from_mat

class Rank300WLPDataset(Dataset):
    def __init__(self, base_dir=None, filename=None, n_class=3, target_size=224, 
        affine_augmenter=None, image_augmenter=None, debug=False, paired_img=False):
        print("[INFO] Initing Rank300WLPDataset.")
        print(base_dir, filename, n_class, target_size, debug)
        self.base_dir = Path(base_dir)
        self.n_class = n_class
        self.target_size = target_size
        self.affine_augmenter = affine_augmenter
        self.image_augmenter = image_augmenter
        self.debug = debug
        self.paired_img = paired_img
        self.CMU_data = False

        self.ids = []
        self.bboxs = []
        self.labels = []
        self.ids_index = []

        with open(self.base_dir/filename) as f:
            print("[INFO] Loading data.")
            for i, line in enumerate(tqdm.tqdm(f.readlines()[:100])):
                ls = line.strip()
                print(ls)

                id_index = []
                bboxs = []
                labels = []

                for j in range(100):
                    img_path = self.base_dir / (ls + '_%d.jpg' % j) if self.paired_img else self.base_dir / (ls)
                    # If CMU dataset
                    if os.path.isfile(img_path):
                      if not os.path.exists(img_path):
                          # if self.debug:
                          #     print(f"[DEBUG] Path not exits: {img_path}")
                          break

                      mat_path = str(img_path).replace('.jpg', '.mat')
                      bbox, pose = get_pt_ypr_from_mat(mat_path)

                      if False and (abs(pose[1])>99 or abs(pose[0])>99 or abs(pose[2])>99):
                          continue

                      id_index.append(j)
                      bboxs.append(bbox)
                      labels.append(np.array(pose))
                    else:
                      # For CMU dataset type
                      # CMU_data/170228_haggling_a2/170228_haggling_a2_cropped/00_2_00001566.jpg,78.16834326232015,61.04268320022404,58.21473740844096

                      img_path = self.base_dir / ls.split(",")[0]

                      if os.path.isfile(img_path):
                        if not self.CMU_data:
                          print("[INFO] Loading CMU dataset")
                        self.CMU_data = True
                        
                      pose = (float(ls.split(",")[1]), float(ls.split(",")[2]), float(ls.split(",")[3]))
                      # Image in CMU dataset is cropped, bbox info not be saved
                      bbox = (0, 0, 0, 0)

                      id_index.append(j)
                      bboxs.append(bbox)
                      labels.append(np.array(pose))
                      
                    if not self.paired_img:
                        break

                # if self.debug:
                #     print(f"[DEBUG] id_index: {id_index}")
                self.labels.append(labels)
                self.bboxs.append(bboxs)
                if not self.CMU_data:
                  self.ids.append(ls)
                else:
                  self.ids.append(self.base_dir / ls.split(",")[0])
                self.ids_index.append(id_index)

                self.resizer = albu.Compose([albu.SmallestMaxSize(target_size, p=1.),
                                        albu.CenterCrop(target_size, target_size, p=1.)])

    def __len__(self):
        return len(self.ids) * 5

    def get_one_img(self, index):
        index = index % len(self.ids)
        # print(f"[INFO] Getting index: {index}")
        # if self.debug:
        #     print(f"[INFO] self.ids_index[index]={self.ids_index}")
        # print(self.ids[index])
        # print(self.bboxs[index])
        # print(self.labels[index])

        img_path1 = self.base_dir / (self.ids[index])
        

        # print(f"[INFO] Path one: {img_path1}")
        # print(f"[INFO] Path two: {img_path2}")

        # scale = np.random.random_sample() * 0.2 + 0.1
        scale = np.random.random_sample() * 0.2 + 1.4
        bbox1 = change_bbox(self.bboxs[index][0], scale=scale, use_forehead=False)
        
        # print(bbox1)
        # Check bbox, if is CMU dataset, bbox = (0, 0, 0, 0)
        if bbox1 == (0, 0, 0, 0):
          img1 = np.array(Image.open(img_path1))
        else:
          img1 = np.array(Image.open(img_path1).crop(bbox1))

        lbl1 = self.labels[index][0]
        # print(f"[INFO] Label: {lbl1}")

        # ImageAugment (RandomBrightness, AddNoise...)
        if self.image_augmenter:
            augmented = self.image_augmenter(image=img1)
            img1 = augmented['image']

        # Resize (Scale & Pad & Crop)
        if self.resizer:
            resized = self.resizer(image=img1)
            img1 = resized['image']
        # AffineAugment (Horizontal Flip, Rotate...)
        if self.affine_augmenter:
            augmented = self.affine_augmenter(image=img1)
            img1 = augmented['image']

        if self.debug:
            # print(label)
            return img1
        else:
            img1 = preprocess(img1)
            img1 = torch.FloatTensor(img1).permute(2, 0, 1)
            lbl1 = torch.FloatTensor(lbl1)

            # print(f"[INFO] Label: {lbl1.shape}")
            
            return img1, lbl1, img1, lbl1, lbl1

    def get_two_imgs(self, index):
        index = index % len(self.ids)
        # print(f"[INFO] Getting index: {index}")
        # if self.debug:
        #     print(f"[INFO] self.ids_index[index]={self.ids_index}")
        idxs = np.random.choice(self.ids_index[index], size=2, replace=False)

        img_path1 = self.base_dir / (self.ids[index]+'_%d.jpg' % idxs[0])
        img_path2 = self.base_dir / (self.ids[index]+'_%d.jpg' % idxs[1])

        # print(f"[INFO] Path one: {img_path1}")
        # print(f"[INFO] Path two: {img_path2}")

        # scale = np.random.random_sample() * 0.2 + 0.1
        scale = np.random.random_sample() * 0.2 + 1.4
        bbox1 = change_bbox(self.bboxs[index][idxs[0]], scale=scale, use_forehead=False)
        bbox2 = change_bbox(self.bboxs[index][idxs[1]], scale=scale, use_forehead=False)
        img1 = np.array(Image.open(img_path1).crop(bbox1))
        img2 = np.array(Image.open(img_path2).crop(bbox2))

        lbl1 = self.labels[index][idxs[0]]
        lbl2 = self.labels[index][idxs[1]]


        # ImageAugment (RandomBrightness, AddNoise...)
        if self.image_augmenter:
            augmented = self.image_augmenter(image=img1)
            img1 = augmented['image']
            augmented = self.image_augmenter(image=img2)
            img2 = augmented['image']

        # Resize (Scale & Pad & Crop)
        if self.resizer:
            resized = self.resizer(image=img1)
            img1 = resized['image']
            resized = self.resizer(image=img2)
            img2 = resized['image']
        # AffineAugment (Horizontal Flip, Rotate...)
        if self.affine_augmenter:
            augmented = self.affine_augmenter(image=img1)
            img1 = augmented['image']
            augmented = self.affine_augmenter(image=img2)
            img2 = augmented['image']

        # label = (lbl1 > lbl2) * 2 - 1
        label = np.sign(lbl1 - lbl2)

        if self.debug:
            # print(label)
            return img1, img2
        else:
            img1 = preprocess(img1)
            img1 = torch.FloatTensor(img1).permute(2, 0, 1)
            img2 = preprocess(img2)
            img2 = torch.FloatTensor(img2).permute(2, 0, 1)

            label = torch.FloatTensor(label.astype(np.float32))
            lbl1 = torch.FloatTensor(lbl1)
            lbl2 = torch.FloatTensor(lbl2)
            
            return img1, img2, lbl1, lbl2, label

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
    dataset = Rank300WLPDataset(base_dir="/home/linhnv/projects/RankPose/data", affine_augmenter=affine_augmenter, image_augmenter=image_augmenter,
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


