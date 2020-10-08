
import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import albumentations as albu
from torch.utils.data import Dataset

import sys
sys.path.append("..")

from utils.preprocess import preprocess, change_bbox
from utils.functional import get_pt_ypr_from_mat

class Rank300WLPDataset(Dataset):
    def __init__(self, base_dir=None, filename=None, n_class=3, target_size=224, affine_augmenter=None, image_augmenter=None, debug=False):
        print("[INFO] Initing Rank300WLPDataset.")
        self.base_dir = Path(base_dir)
        self.n_class = n_class
        self.target_size = target_size
        self.affine_augmenter = affine_augmenter
        self.image_augmenter = image_augmenter
        self.debug = debug

        self.ids = []
        self.bboxs = []
        self.labels = []
        self.ids_index = []

        with open(self.base_dir/filename) as f:
            for i, line in enumerate(f.readlines()):
                ls = line.strip()

                id_index = []
                bboxs = []
                labels = []

                for j in range(100):
                    img_path = self.base_dir / (ls + '_%d.jpg' % j)
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

                # if self.debug:
                #     print(f"[DEBUG] id_index: {id_index}")
                self.labels.append(labels)
                self.bboxs.append(bboxs)
                self.ids.append(ls)
                self.ids_index.append(id_index)

                self.resizer = albu.Compose([albu.SmallestMaxSize(target_size, p=1.),
                                        albu.CenterCrop(target_size, target_size, p=1.)])

    def __len__(self):
        return len(self.ids) * 5

    def __getitem__(self, index):
        index = index % len(self.ids)
        # if self.debug:
        #     print(f"[INFO] self.ids_index[index]={self.ids_index}")
        idxs = np.random.choice(self.ids_index[index], size=2, replace=False)

        img_path1 = self.base_dir / (self.ids[index]+'_%d.jpg' % idxs[0])
        img_path2 = self.base_dir / (self.ids[index]+'_%d.jpg' % idxs[1])

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


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    affine_augmenter = None
    image_augmenter = albu.Compose([albu.GaussNoise((0, 25), p=.5),
                                    albu.RandomBrightnessContrast(0.4, 0.3, p=1),
                                    albu.JpegCompression(90, 100, p=0.5)])
    #image_augmenter = None
    image_augmenter = albu.Compose([albu.RandomBrightnessContrast(0.4,0.3,p=0.5),
                                    albu.RandomGamma(p=0.3),
                                    albu.CLAHE(p=0.1),
                                    albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20,p=0.2),
                                    ])
    dataset = Rank300WLPDataset(base_dir="/home/linhnv/projects/RankPose/data", affine_augmenter=affine_augmenter, image_augmenter=image_augmenter,
                             filename='300w_lp_for_rank.txt', target_size=64, debug=True)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(len(dataset))

    for i, batched in enumerate(dataloader):
        img1, img2 = batched
        for j in range(8):
            img = img1[j].numpy()
            img = img.astype('uint8')
            img = Image.fromarray(img)
            img.save('300W_img1_%d_%d.jpg'%(i, j))
            img = img2[j].numpy()
            img = img.astype('uint8')
            img = Image.fromarray(img)
            img.save('300W_img2_%d_%d.jpg'%(i, j))
        if i > 2:
            break


