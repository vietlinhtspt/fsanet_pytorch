import math
import os
import yaml
import json
from glob import glob
import sys
from pathlib import Path
from PIL import Image
from torchvision import transforms
import tqdm
import cv2
from math import cos, sin
import albumentations as albu
import numpy as np
import torch
import torch.nn as nn
from models import load_model

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
            cropped_frame = resizer(image=cropped_frame)
            # Convert to tensor
            cropped_frame = Image.fromarray(cropped_frame['image'])
            cropped_frame = transforms.ToTensor()(cropped_frame).unsqueeze_(0)
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

    

if __name__ == "__main__":
    dir_imgs = "/media/2tb/projects/VL's/headpose_data/CCTV_Shophouse/raw_frames"
    label_box_imgs = "/media/2tb/projects/VL's/headpose_data/CCTV_Shophouse/labeled_faces"
    model_path = "/media/2tb/projects/VL's/FSANet/models/fsanet_Wrapped_HDF5_CMU_300WLP/model_epoch_89_11.195596277713776.pth"
    config_model_path = "/home/linhnv/projects/fsanet_pytorch/config/fsanet_wrapped_colab_CMU.yaml"

    save_dir = "/media/2tb/projects/VL's/headpose_data/CCTV_Shophouse"

    # data_name = os.path.basename(dir_imgs)
    data_name = "CCTV_Shophouse_fsanet_wrapped"
    save_path = os.path.join(save_dir, data_name)
    
    save_imgs_path = os.path.join(save_path, "processed_imgs")
    save_video_path = os.path.join(save_path, "processed_videos")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(save_imgs_path).mkdir(parents=True, exist_ok=True)
    Path(save_video_path).mkdir(parents=True, exist_ok=True)

    config = yaml.load(open(config_model_path))
    net_config = config['Net']
    model = load_model(**net_config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")
    # To device
    model = model.to(device)

    if torch.cuda.is_available():
        model = nn.DataParallel(model)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        visualize(model, dir_imgs, label_box_imgs, save_imgs_path=save_imgs_path, save_video_path=save_video_path)
    