import h5py
import os
import numpy as np
from PIL import Image
import cv2
import pickle
import tqdm
import albumentations as albu

import sys
sys.path.append("..")

from utils.preprocess import preprocess, change_bbox
from utils.functional import get_pt_ypr_from_mat

class HDF5DatasetWriter:
    def __init__(self, dims, outputPath, dataKey="images", bufSize=400):
        # check to see if the output path exists, and if so, raise
        # an exception
        if os.path.exists(outputPath):
            raise ValueError("The supplied ‘outputPath‘ already exists and cannot be overwritten. Manually delete the file before continuing.", outputPath)

        # open the HDF5 database for writing and create two datasets:
        # one to store the images/features and another to store the
        # class labels
        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset(dataKey, dims, dtype=np.uint8)
        self.size_imgs = self.db.create_dataset("raw_size_data", (dims[0], 3) , dtype=np.uint8)
        self.labels = self.db.create_dataset("labels", (dims[0], 3), dtype="float")
        self.size_data = (dims[1], dims[2], dims[3])

        # self.resizer = albu.Compose([albu.SmallestMaxSize(target_size, p=1.), albu.CenterCrop(target_size, target_size, p=1.)])
        # store the buffer size, then initialize the buffer itself
        # along with the index into the datasets
        self.bufSize = bufSize
        self.buffer = {"data": [], "raw_size_data": [], "labels": []}
        self.idx = 0

    def add(self, rows, label):
        # add the rows and labels to the buffer
        
        padded_data = np.zeros(self.size_data)
        padded_data[:rows.shape[0],:rows.shape[1]] = rows
        size_raw_data = np.array(rows.shape)
        # print("Size label: ", label)
        # print("Size img: ", size_raw_data)
        
        self.buffer["data"].extend([padded_data])
        self.buffer["raw_size_data"].extend([size_raw_data])
        self.buffer["labels"].extend([label])
        
        # check to see if the buffer needs to be flushed to disk
        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()
    
    def flush(self):
        # write the buffers to disk then reset the buffer
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        # self.size_imgs[self.idx:i] = self.buffer["raw_size_data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "raw_size_data": [], "labels": []}

    def close(self):
        # check to see if there are any other entries in the buffer
        # that need to be flushed to disk
        if len(self.buffer["data"]) > 0:
            self.flush()

        # close the dataset
        self.db.close()

if __name__ == "__main__":

    img_paths = []
    labels = []
    bboxs = []
    dir_path = "/home/linhnv/projects/RankPose/data"

    with open('/home/linhnv/projects/fsanet_pytorch/training_and_testing/cache_rank_300w_lp.pickle', 'rb') as handle:
        pickle_data = pickle.load(handle)
        img_paths = pickle_data[0]
        labels = pickle_data[2]
        bboxs = pickle_data[1]

    path_save_HDF5_file = f"/media/2tb/projects/VL's/FSANet/300W_LP_dataset_{len(img_paths)}.hdf5"
    
    # img_shapes = []

    os.system(f"rm -rf {path_save_HDF5_file}")

    writer = HDF5DatasetWriter((len(img_paths), 450, 450, 3), path_save_HDF5_file)

    for id, path_img in enumerate(tqdm.tqdm(img_paths[:])):
        # print(path_img)
        img_path = dir_path + "/" + path_img.split(",")[0]

        scale = np.random.random_sample() * 0.2 + 1.4
        # print(bboxs[id])
        bbox = change_bbox(bboxs[id][0], scale=scale, use_forehead=False)

        img = np.array(Image.open(img_path).crop(bbox))
        # padded_img = np.zeros((1080, 1917, 3))
        # padded_img[:img.shape[0],:img.shape[1]] = img

        # print(img)

        # label = np.array([float(line.split(",")[1]), float(line.split(",")[2]), float(line.split(",")[3])])
        # print(labels[id])
        label = np.array(labels[id])
        # print(label)

        writer.add(img,label.reshape(-1))


    # if img.shape not in img_shapes:
    #   img_shapes.append(img.shape)

    # print(img_shapes)
    writer.close()
