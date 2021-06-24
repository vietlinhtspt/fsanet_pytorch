import math
import os
import yaml
import json
from glob import glob
import sys
from pathlib import Path
from PIL import Image
from torchvision import transforms
from utils.preprocess import preprocess, change_bbox
from utils.functional import get_pt_ypr_from_mat
import tqdm
import cv2
from math import cos, sin
import albumentations as albu
import numpy as np
import torch
import h5py
import torch.nn as nn
from models import load_model
from training_and_testing.validate import create_model, load_pretrain_model

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=80):
    """
    input:
        img: cv2 image
        yaw: euler (360) 
        pitch: euler (360)
        roll: euler (360)
        tdx, tdy: tdx = width / 2, tdy = height / 2
    output:
    """
    yaw_euler = yaw
    pitch_euler = pitch
    roll_euler = roll
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll)
                 * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch)
                 * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

    # cv2.putText(img, "yaw"+str(yaw_euler), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
    # cv2.putText(img, "pitch"+str(pitch_euler), (int(x2), int(y2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
    # cv2.putText(img, "roll"+str(roll_euler), (int(x3), int(y3)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

    return img

def draw_annotate(image, bbox, yaw, pitch, roll):
    """
    input:
        image: RGB image
        bbox: np.array([x1_min,y1_min, x2_max, y2_max])
        pose: yaw, pitch, roll
    output:
        draw bbox and pose on image
    """
    x, y = bbox[:2]
    w, h = (bbox[2:] - bbox[:2])
    x, y, w, h = int(x), int(y), int(w), int(h)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,255), 2)
    x_c, y_c = int(x + w / 2), int(y + h / 2)
    draw_axis(image, yaw, pitch, roll, x_c, y_c)
    # cv2.imwrite("test.jpg", image)
    return image

def draw_annotates(img, bboxs, poses):
    """
    input:
        image: RGB image
        bboxs: [[x1_min,y1_min, x2_max, y2_max]]
        poses: [[yaw, pitch, roll]]
    output:
        draw bboxs and poses on image
    """
    for index, bbox in enumerate(bboxs):
    # print(f"[INFO] bbox: {bboxs}")
        draw_annotate(img, np.array([bbox[0], bbox[1], bbox[2], bbox[3]]), poses[index][0], poses[index][1], poses[index][2])  
    return img

def get_bbox_YOLO_format(label_path, width, height):
    """
    input:
        label_path: path to label with YOLO format
        width: width img
        height: height img
    output:
        bboxs in label file
    """
    bboxs_output = []
    with open(label_path) as f:
        lines = f.readlines()
        # "0 0.341095 0.475086 0.032063 0.063574\n"
        bboxs = [line[:-2].split(" ")[1:] for line in lines]
        for bbox in bboxs[:]:
            # print(bbox)
            # print(height, width)
            x1 = float(bbox[0]) * width 
            y1 = float(bbox[1]) * height 
            width_bbox = float(bbox[2]) * width 
            height_bbox = float(bbox[3]) * height 
            x1 = x1 - int(width_bbox / 2)
            y1 = y1 - int(height_bbox / 2)
            # print(x1, y1, width, height)
            bboxs_output.append([x1, y1, x1+width_bbox, y1+height_bbox])

    return bboxs_output


def read_frames(dir_imgs):
    # print(os.listdir(dir_imgs))
    images = [os.path.join(dir_imgs, img) for img in os.listdir(dir_imgs) 
              if img.endswith(".jpg") or
                 img.endswith(".jpeg") or
                 img.endswith("png")] 
    path_images = sorted(images, key=lambda x: float(os.path.basename(x).split(".")[0]), reverse=False)
    
    return path_images

def visualize(model, dir_imgs, label_box_imgs, save_imgs_path=None, save_video_path=None, target_size_imgs = 64,):
    """
    input:
        dir_imgs: path to saved imgs directory
        label_box_imgs: path to saved bbox directory
    output:
        draw bboxs and poses on images.
        Export to imgs and video(option)
    """

    list_imgs = read_frames(dir_imgs)
    resizer = albu.Compose([albu.SmallestMaxSize(target_size_imgs, p=1.), albu.CenterCrop(target_size_imgs, target_size_imgs, p=1.)])
    # head_pose_offset = [head_pose_init[0] - camera_yaw, head_pose_init[1] - camera_pitch, head_pose_init[2] - camera_roll]
    
    print("[INFO] Drawing ...")
    for index, img_path in enumerate(tqdm.tqdm(list_imgs[:])):
        # print(img_path)
        img = cv2.imread(img_path)
        height, width, channels = img.shape

        # print(height, width)
        img_name = os.path.basename(img_path)
        if img_name.split(".")[-1] == "jpg":
            label_name = img_name.replace(".jpg", ".txt")
        else:
            label_name = img_name.replace(".png", ".txt")
        label_path = os.path.join(label_box_imgs, label_name)
        bboxs = get_bbox_YOLO_format(label_path, width, height)
        poses = []
        labeled_bboxs = []
        for bbox in bboxs:
            bbox = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])
            x, y = bbox[:2]
            w, h = (bbox[2:] - bbox[:2])
            x, y, w, h = int(x), int(y), int(w), int(h)
            cropped_frame = img[y:y+h, x:x+w]
            # Todo: resize frame. If size < 64 then scale up else scale down 
            # but both methods keep aspect ratio.
            if w < target_size_imgs or h < target_size_imgs:
                scale_percent = target_size_imgs / min(w, h)
                width = int(img.shape[1] * scale_percent)
                height = int(img.shape[0] * scale_percent)
            
            # cv2.imwrite("cropped_frame.jpg", cropped_frame) 
            # Scale down and crop to target size
            if cropped_frame.shape[0] == 0 or cropped_frame.shape[1] == 0:
                continue
            cropped_frame = resizer(image=cropped_frame)
           
            cropped_frame = preprocess(np.array(cropped_frame['image']))
            # print(cropped_frame)
            # Convert to tensor and transform to [-1, 1]
            cropped_frame = torch.FloatTensor(cropped_frame).permute(2, 0, 1).unsqueeze_(0)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
            cropped_frame.to(device)
            # print(cropped_frame.size())
            # predict
            preds = model(cropped_frame)
            preds = preds.cpu().detach().numpy()
            
            poses.append(preds[0])
            labeled_bboxs.append(bbox)
        # print(bboxs)
        poses = np.array(poses)
        labeled_bboxs = np.array(labeled_bboxs)
        # print(poses)
        drawed_img = draw_annotates(img, labeled_bboxs, poses)
        if save_imgs_path:
            cv2.imwrite(os.path.join(save_imgs_path, img_name), drawed_img)

        # print(img_name)
        # print(f"[INFO] Head sensor: [{head_poses['yaw']}, {head_poses['pitch']}, {head_poses['pitch']}]")
        # print(f"[INFO] Head init: [{head_pose_init[0], head_pose_init[1], head_pose_init[2]}]")
        # # print(f"[INFO] Camera sensor: [{camera_yaw}, {camera_pitch}, {camera_roll}]")
        # print(f"[INFO] Img pose: [{head_yaw}, {head_pitch}, {head_roll}]")
        
        # print(label_path)
    # print(len(list_imgs))
    # print(len(list_label_boxs))
    if save_video_path:
        generate_video(save_imgs_path, save_video_path)

def visualize_fusion(uni_model, var_model, wei_model, dir_imgs, label_box_imgs, save_imgs_path=None, save_video_path=None, save_cropped_path=None, target_size_imgs = 64):
    """
    input:
        uni_model:
        var_model:
        wei_model:
        dir_imgs: path to save imgs directory
        label_box_imgs: path to save bbox directory
        save_cropped_path: path to save cropped imgs directory
    output:
        draw bboxs and poses on images.
        Export to imgs and video(option)
    """

    list_imgs = read_frames(dir_imgs)
    resizer = albu.Compose([albu.SmallestMaxSize(target_size_imgs, p=1.), albu.CenterCrop(target_size_imgs, target_size_imgs, p=1.)])
    # head_pose_offset = [head_pose_init[0] - camera_yaw, head_pose_init[1] - camera_pitch, head_pose_init[2] - camera_roll]
    print("[INFO] Drawing ...")
    for index, img_path in enumerate(tqdm.tqdm(list_imgs[:])):
        # print(img_path)
        img = cv2.imread(img_path)
        height, width, channels = img.shape

        # print(height, width)
        img_name = os.path.basename(img_path)
        if img_name.split(".")[-1] == "jpg":
            label_name = img_name.replace(".jpg", ".txt")
        else:
            label_name = img_name.replace(".png", ".txt")
        label_path = os.path.join(label_box_imgs, label_name)
        bboxs = get_bbox_YOLO_format(label_path, width, height)
        poses = []
        labeled_bboxs = []
        for index in range(0, len(bboxs)):
            bbox = bboxs[index]
            bbox = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])
            x, y = bbox[:2]
            w, h = (bbox[2:] - bbox[:2])
            x, y, w, h = int(x), int(y), int(w), int(h)
            cropped_frame = img[y:y+h, x:x+w]
            # Todo: resize frame. If size < 64 then scale up else scale down 
            # but both methods keep aspect ratio.
            if w < target_size_imgs or h < target_size_imgs:
                scale_percent = target_size_imgs / min(w, h)
                width = int(img.shape[1] * scale_percent)
                height = int(img.shape[0] * scale_percent)
            
            # cv2.imwrite("cropped_frame.jpg", cropped_frame) 
            # Scale down and crop to target size
            if cropped_frame.shape[0] == 0 or cropped_frame.shape[1] == 0:
                continue
            cropped_frame = resizer(image=cropped_frame)
            cropped_frame_copy = np.copy(np.array(cropped_frame['image']))
           
            cropped_frame = preprocess(np.array(cropped_frame['image']))
            # print(cropped_frame)
            # Convert to tensor and transform to [-1, 1]
            cropped_frame = torch.FloatTensor(cropped_frame).permute(2, 0, 1).unsqueeze_(0)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
            cropped_frame.to(device)
            # print(cropped_frame.size())
            # predict
            
            uni_preds = uni_model(cropped_frame)
            var_preds = var_model(cropped_frame)
            wei_preds = wei_model(cropped_frame)

            preds = uni_preds.add(var_preds).add(wei_preds)/3
            preds = preds.cpu().detach().numpy()
            if save_cropped_path:
                
                drawed_cropped_img = draw_annotates(cropped_frame_copy, np.array([[0, 0, 63, 63]]), np.array([preds[0]]))
                absolute_name_img = img_name.split(".")[0]
                
                cropped_img_name = f"{absolute_name_img}_{index}.jpg"
                # print(os.path.join(save_cropped_path, cropped_img_name))
                cv2.imwrite(os.path.join(save_cropped_path, cropped_img_name), drawed_cropped_img)
            
            poses.append(preds[0])
            labeled_bboxs.append(bbox)
        # print(bboxs)
        poses = np.array(poses)
        labeled_bboxs = np.array(labeled_bboxs)
        # print(poses)
        drawed_img = draw_annotates(img, labeled_bboxs, poses)
        if save_imgs_path:
            cv2.imwrite(os.path.join(save_imgs_path, img_name), drawed_img)

        # print(img_name)
        # print(f"[INFO] Head sensor: [{head_poses['yaw']}, {head_poses['pitch']}, {head_poses['pitch']}]")
        # print(f"[INFO] Head init: [{head_pose_init[0], head_pose_init[1], head_pose_init[2]}]")
        # # print(f"[INFO] Camera sensor: [{camera_yaw}, {camera_pitch}, {camera_roll}]")
        # print(f"[INFO] Img pose: [{head_yaw}, {head_pitch}, {head_roll}]")
        
        # print(label_path)
    # print(len(list_imgs))
    # print(len(list_label_boxs))
    if save_video_path:
        generate_video(save_imgs_path, save_video_path)

def visualize_HDF5(model, HDF5_path, save_imgs_path=None, save_video_path=None, target_size_imgs = 64,):
    data = h5py.File(HDF5_path)

    print("[INFO] Drawing ...")
    for index in tqdm.tqdm(range(0, 2000)):
        # print(img_path)
        img = np.copy(data["images"][index][:,:,::-1])
        height, width, channels = img.shape

        # print(height, width)
        poses = []
        labeled_bboxs = []
        
        cropped_frame = preprocess(np.array(img))
        # print(cropped_frame)
        # Convert to tensor and transform to [-1, 1]
        cropped_frame = torch.FloatTensor(cropped_frame).permute(2, 0, 1).unsqueeze_(0)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        cropped_frame.to(device)
        # print(cropped_frame.size())
        # predict
        preds = model(cropped_frame)
        preds = preds.cpu().detach().numpy()
        poses.append(preds[0])
        labeled_bboxs.append([0, 0, 63, 63])
        # print(bboxs)
        poses = np.array(poses)
        labeled_bboxs = np.array(labeled_bboxs)
        # print(poses)
        drawed_img = draw_annotates(img, labeled_bboxs, poses)
        img_name = f"{index}.jpg"
        if save_imgs_path:
            cv2.imwrite(os.path.join(save_imgs_path, img_name), drawed_img)

        # print(img_name)
        # print(f"[INFO] Head sensor: [{head_poses['yaw']}, {head_poses['pitch']}, {head_poses['pitch']}]")
        # print(f"[INFO] Head init: [{head_pose_init[0], head_pose_init[1], head_pose_init[2]}]")
        # # print(f"[INFO] Camera sensor: [{camera_yaw}, {camera_pitch}, {camera_roll}]")
        # print(f"[INFO] Img pose: [{head_yaw}, {head_pitch}, {head_roll}]")
        
        # print(label_path)
    # print(len(list_imgs))
    # print(len(list_label_boxs))
    if save_video_path:
        generate_video(save_imgs_path, save_video_path)


# Video Generating function 
def generate_video(imgs_path, saved_video_path):
    """
    input:
        imgs_path: Path to dir saved all drawed imgs
        saved_video_path: Saving video path
    output:
    """
    print(f"[INFO] Wrting video.")
    # print(imgs_path)
    images = read_frames(imgs_path)
    
    # print(images)
    frame = cv2.imread(images[0]) 
    # print(frame.shape)
   
  
    # setting the frame width, height width 
    # the width, height of first image 
    height, width, layers = frame.shape   
  
    video = cv2.VideoWriter(os.path.join(saved_video_path, "output.avi"), 0, 5, (width, height))  
  
    # Appending the images to the video one by one 
    for image in tqdm.tqdm(images):  
        video.write(cv2.imread(os.path.join(imgs_path, image)))  
      
    # Deallocating memories taken for window creation 
    cv2.destroyAllWindows()  
    video.release()  # releasing the video generated 

def visualize_AFLW2000(model, base_dir, target_size, filename, save_imgs_path):
    print("[INFO] Initing ALFW2000Dataset.")
    base_dir = Path(base_dir)
    target_size = target_size

    img_paths = []
    bboxs = []
    labels = []
    pred_poses = []


    with open(base_dir / filename) as f:
        for i, line in enumerate(tqdm.tqdm(f.readlines()[:])):
            ls = line.strip()

            mat_path = base_dir / ls.replace('.jpg', '.mat')
            bbox, pose = get_pt_ypr_from_mat(mat_path, pt3d=True)

            if True and (abs(pose[0])>99 or abs(pose[1])>99 or abs(pose[2])>99):
                continue

            labels.append(np.array(pose))
            bboxs.append(bbox)
            img_paths.append(ls)

    resizer = albu.Compose([albu.SmallestMaxSize(target_size, p=1.),
                                        albu.CenterCrop(target_size, target_size, p=1.)])

    for index in tqdm.tqdm(range(0, len(img_paths[:]))):
        img_path = base_dir /  img_paths[index]
        img_name = os.path.basename(img_paths[index])
        # print(img_path)
        bbox = change_bbox(bboxs[index], 2, use_forehead=False)
        # bbox = bboxs[index]
        raw_img = Image.open(img_path)
        img = np.array(raw_img.crop(bbox))
        img = img[:,:,::-1]

        img = resizer(image=img)
        img = preprocess(np.array(img['image']))
        # print(img)
        # Convert to tensor and transform to [-1, 1]
        img = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze_(0)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        img.to(device)
        

        pred = model(img)
        pred = pred.cpu().detach().numpy()
        pred_poses.append(pred[0])

        if save_imgs_path:
            # print(bbox)
            # print(pred[0])
            b, g, r = raw_img.split()
            raw_img = Image.merge("RGB", (r, g, b))
            drawed_img = draw_annotates(np.array(raw_img), np.array([bbox]), np.array([pred[0]]))
            cv2.imwrite(os.path.join(save_imgs_path, img_name), drawed_img)

    pred_poses = np.array(pred_poses)  
    labels = np.array(labels)

    delta = np.absolute(labels - pred_poses)
    delta = np.sum(delta, axis=1) / 3
    # print(delta)

    sortted_delta = np.sort(delta)
    # print(sortted_delta)
    index_of_max_delta = [np.where(delta==value_delta) for value_delta in sortted_delta[-5:]]
    index_of_min_delta = [np.where(delta==value_delta) for value_delta in sortted_delta[:5]]

    max_delta_dir = os.path.join(save_imgs_path, "max_delta")
    os.system(f"rm -rf \"{max_delta_dir}\"")

    min_delta_dir = os.path.join(save_imgs_path, "min_delta")
    os.system(f"rm -rf \"{min_delta_dir}\"")

    # print(index_of_max_delta)
    for index in index_of_max_delta:
        # print(index[0][0])
        Path(max_delta_dir).mkdir(parents=True, exist_ok=True)

        img_name = os.path.basename(img_paths[index[0][0]])
        path_drawed_img = os.path.join(save_imgs_path, img_name)

        os.system(f"cp -i \"{path_drawed_img}\" \"{max_delta_dir}\"")

    for index in index_of_min_delta:
        # print(index[0][0])

        Path(min_delta_dir).mkdir(parents=True, exist_ok=True)

        img_name = os.path.basename(img_paths[index[0][0]])
        path_drawed_img = os.path.join(save_imgs_path, img_name)

        os.system(f"cp -i \"{path_drawed_img}\" \"{min_delta_dir}\"")


if __name__ == "__main__":
    # dir_imgs = "/media/2tb/projects/VL's/headpose_data/CCTV_Shophouse/raw_frames"
    # label_box_imgs = "/media/2tb/projects/VL's/headpose_data/CCTV_Shophouse/labeled_faces"
    # config_model_path = "/home/linhnv/projects/fsanet_pytorch/config/fsanet_wrapped_colab_CMU.yaml"

    # dir_imgs = "/media/2tb/projects/VL's/headpose_data/CMU_28_10s/raw_frames"
    # label_box_imgs = "/media/2tb/projects/VL's/headpose_data/CMU_28_10s/labeled_faces"
    # model_path = "/media/2tb/projects/VL's/FSANet/models/fsanet_MSE_HDF5_CMU_300WLP/model_epoch_88_10.852409601211548.pth"
    # config_model_path = "/home/linhnv/projects/fsanet_pytorch/config/fsanet_wrapped_colab_CMU.yaml"

    # dir_imgs = "/media/2tb/projects/VL's/headpose_data/CMU_23_30s/raw_frames"
    # label_box_imgs = "/media/2tb/projects/VL's/headpose_data/CMU_23_30s/labeled_faces"
    # model_path = "/media/2tb/projects/VL's/FSANet/models/fsanet_MSE_HDF5_CMU_300WLP/model_epoch_88_10.852409601211548.pth"
    # config_model_path = "/home/linhnv/projects/fsanet_pytorch/config/fsanet_wrapped_colab_CMU.yaml"

    dir_imgs = "/media/2tb/projects/VL's/headpose_data/AILab_mask_nomask/raw_frames"
    label_box_imgs = "/media/2tb/projects/VL's/headpose_data/AILab_mask_nomask/labeled_faces"
    # model_path = "/media/2tb/projects/VL's/FSANet/models/fsanet_MSE_HDF5_CMU_300WLP/model_epoch_88_10.852409601211548.pth"
    config_model_path = "/home/linhnv/projects/fsanet_pytorch/config/fsanet_wrapped_colab_CMU.yaml"

    # dir_imgs = "/media/2tb/projects/VL's/headpose_data/CMU_23_30s/raw_frames"
    # label_box_imgs = "/media/2tb/projects/VL's/headpose_data/CMU_23_30s/CMU_23_30s_labeled_faces_lech_duoi"
    # model_path = "/media/2tb/projects/VL's/FSANet/models/fsanet_MSE_HDF5_top_CMU_300WLP/model_epoch_98_11.402141273021698.pth"
    # config_model_path = "/home/linhnv/projects/fsanet_pytorch/config/fsanet_wrapped_colab_CMU.yaml"

    # dir_imgs = "/media/2tb/projects/VL's/headpose_data/Demo_office_JP_CX_18s/raw_frames"
    # label_box_imgs = "/media/2tb/projects/VL's/headpose_data/Demo_office_JP_CX_18s/labeled_faces"
    # model_path = "/media/2tb/projects/VL's/FSANet/models/fsanet_MSE_HDF5_CMU_300WLP/model_epoch_88_10.852409601211548.pth"
    # config_model_path = "/home/linhnv/projects/fsanet_pytorch/config/fsanet_wrapped_colab_CMU.yaml"



    # save_imgs_AFLW2000_path = "/media/2tb/projects/VL's/headpose_data/drawed_AFLW2000"

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

    val_dir = "/media/2tb/projects/VL's/UetHeadpose/pre_processed/val_data/"
    val_name = "CMU_dataset_64x64"
    val_type = "HDF5_multi"

    save_dir = "/media/2tb/projects/VL's/headpose_data/AILab_mask_nomask"

    # data_name = os.path.basename(dir_imgs)
    data_name = "fsa_MSE_CMU_fusion"
    save_path = os.path.join(save_dir, data_name)
    
    save_imgs_path = os.path.join(save_path, "processed_imgs")
    save_cropped_path = os.path.join(save_path, "cropped_imgs")
    save_video_path = os.path.join(save_path, "processed_videos")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(save_imgs_path).mkdir(parents=True, exist_ok=True)
    Path(save_cropped_path).mkdir(parents=True, exist_ok=True)
    Path(save_video_path).mkdir(parents=True, exist_ok=True)
    # Path(save_imgs_AFLW2000_path).mkdir(parents=True, exist_ok=True)

    config = yaml.load(open(var_config_path))
    net_config = config['Net']
    model = load_model(**net_config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")
    # To device
    model = model.to(device)

    if torch.cuda.is_available():
        model = nn.DataParallel(model)

    uni_model, target_size, num_workers, batch_size, n_class = create_model(uni_config_path)
    var_model, target_size, num_workers, batch_size, n_class = create_model(var_config_path)
    wei_model, target_size, num_workers, batch_size, n_class = create_model(wei_config_path)

    uni_model = load_pretrain_model(uni_model, uni_model_path)
    var_model = load_pretrain_model(var_model, var_model_path)
    wei_model = load_pretrain_model(wei_model, wei_model_path)

    valid_diffs = []

    uni_model.eval()
    var_model.eval()
    wei_model.eval()

    model.load_state_dict(torch.load(var_model_path)["model_state_dict"])
    model.eval()

    with torch.no_grad():
        # visualize(model, dir_imgs, label_box_imgs, save_imgs_path=save_imgs_path, save_video_path=save_video_path)
        # visualize(model, dir_imgs, label_box_imgs, save_imgs_path=save_imgs_path)
        # visualize_HDF5(model, "/media/2tb/projects/VL's/UetHeadpose/pre_processed/val_data_01/UETHeadpose_val_64x64_0_2000.hdf5", "/media/2tb/projects/VL's/UetHeadpose/pre_processed/val_data/pred_val_imgs")

        visualize_fusion(uni_model, var_model, wei_model, dir_imgs, label_box_imgs, save_imgs_path=save_imgs_path, save_cropped_path=save_cropped_path)

        # visualize_AFLW2000(model, "/media/2tb/projects/VL's/headpose_data", 64, "aflw2000_filename.txt", save_imgs_AFLW2000_path)

    generate_video("/media/2tb/projects/VL's/headpose_data/AILab_mask_nomask/fsa_MSE_CMU_fusion/cropped_imgs", "/home/linhnv/projects/fsanet_pytorch")