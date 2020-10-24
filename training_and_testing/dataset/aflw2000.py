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

class AFLW2000Dataset(Dataset):
    def __init__(self, base_dir=None, filename=None, n_class=3, target_size=224, 
        affine_augmenter=None, image_augmenter=None, debug=False):
        print("[INFO] Initing ALFW2000Dataset.")
        self.base_dir = Path(base_dir)
        self.n_class = n_class
        self.target_size = target_size
        self.affine_augmenter = affine_augmenter
        self.image_augmenter = image_augmenter
        self.debug = debug

        self.img_paths = []
        self.bboxs = []
        self.labels = []
        
        # Read img, bbox, pose
        with open(self.base_dir / filename) as f:
            for i, line in enumerate(f.readlines()):
                ls = line.strip()

                mat_path = self.base_dir / ls.replace('.jpg', '.mat')
                bbox, pose = get_pt_ypr_from_mat(mat_path, pt3d=True)

                if True and (abs(pose[0])>99 or abs(pose[1])>99 or abs(pose[2])>99):
                    continue

                self.labels.append(np.array(pose))
                self.bboxs.append(bbox)
                self.img_paths.append(ls)

        # labels_sort_idx is created by the magnitude of 3 args (yall, pitch, row)
        self.labels_sort_idx = np.argsort(-np.mean(np.abs(self.labels), axis=1))
        self.resizer = albu.Compose([albu.SmallestMaxSize(target_size, p=1.),
                                        albu.CenterCrop(target_size, target_size, p=1.)])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.base_dir /  self.img_paths[index]
        bbox = change_bbox(self.bboxs[index], 1.4, use_forehead=False)
        img = np.array(Image.open(img_path).crop(bbox))

        label = self.labels[index].copy()

        # ImageAugment (RandomBrightness, AddNoise...)
        if self.image_augmenter:
            augmented = self.image_augmenter(image=img)
            img = augmented['image']

        # Resize (Scale & Pad & Crop)
        if self.resizer:
            resized = self.resizer(image=img)
            img = resized['image']
        # AffineAugment (Horizontal Flip, Rotate...)
        if self.affine_augmenter:
            augmented = self.affine_augmenter(image=img)
            img = augmented['image']

        if self.debug:
            print(self.bboxs[index])
            print(label)
        else:
            img = preprocess(img)
            img = img.transpose(2, 0, 1)
            img = torch.FloatTensor(img)
            label = torch.FloatTensor(label)
        return img, label

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
    dataset = AFLW2000Dataset(base_dir="/home/linhnv/projects/RankPose/data", affine_augmenter=affine_augmenter, image_augmenter=image_augmenter,
                             filename='aflw2000_filename.txt', target_size=224, debug=True)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    print(len(dataset))

    for i, batched in enumerate(dataloader):
        images, labels = batched
        print(images.shape)
        # for j in range(8):
        #     img = images[j].numpy()
        #     img = img.astype('uint8')
        #     img = Image.fromarray(img)
        #     img.save('tmp/%d_%d.jpg'%(i, j))
        # # if i > 2:
        # #     break
