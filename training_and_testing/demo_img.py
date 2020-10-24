from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import yaml
import os
import torch
import argparse
import torch.nn as nn
from models import load_model
from torchvision import transforms
from dataset.draw import draw_axis

def crop_and_pred(img_path, bboxes, model, device, target_size, scale=1):
    print(img_path)
    img = cv2.imread(img_path)
    scale_percent = scale*100

    #calculate the 50 percent of original dimensions
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)

    # dsize
    dsize = (width, height)

    # resize image
    # img = cv2.resize(img, dsize)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    model.eval()
    with torch.no_grad():
        for bbox in bboxes:
            print(bbox)
            x_min, y_min, x_max, y_max = bbox
            x_min, y_min, x_max, y_max = x_min * 2, y_min * 2, x_max * 2, y_max * 2
            print(f"[INFO] W, H = {y_max - y_min}:{x_max - x_min}")
            img_face_rgb = img_rgb[y_min:y_max, x_min:x_max]
            img_face_rgb = cv2.resize(img_face_rgb,(target_size,target_size))
            img_face_rgb = np.expand_dims(img_face_rgb, axis=0)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0,0,0), 1)

            # print(img_face_rgb[0].shape)
            img_face_rgb = Image.fromarray(img_face_rgb[0])
            img_face_rgb = transforms.ToTensor()(img_face_rgb).unsqueeze_(0)
            img_face_rgb = img_face_rgb.to(device)
            
            output = model(img_face_rgb).cpu().numpy()[0] # tensor([[  9.1769, -28.2790,  -0.1838]], device='cuda:0')
            #  [yaw, pitch, roll]
            print("yaw: {yaw}, pitch: {pitch}, row: {row}")
            draw_axis(img, output[0], output[1], output[2], tdx=(x_min+x_max)/2, tdy=(y_min+y_max)/2, size = abs(x_max-x_min))

            img_face_rgb = img[y_min:y_max, x_min:x_max]
            img_face_rgb = cv2.cvtColor(img_face_rgb, cv2.COLOR_BGR2RGB)
            img_face_rgb = Image.fromarray(img_face_rgb)
            file_name = str(os.path.basename(img_path)).split(".")[3]
            img_face_rgb.save(Path("/home/linhnv/projects/fsanet_pytorch/demo/rankpose/cropped_imgs") / f"{file_name}_{y_max-y_min}_{x_max-x_min}.jpg")

    # cv2.imshow(os.path.basename(img_path),img)
    img = img[:,:,::-1]
    img = cv2.resize(img, dsize)
    img = Image.fromarray(img)
    img.save(f"{os.path.basename(img_path)}")
    # cv2.waitKey(0)

parser = argparse.ArgumentParser()
parser.add_argument('config_path')
args = parser.parse_args()

if __name__ == "__main__":
    config_path = Path(args.config_path)
    config = yaml.load(open(config_path))
    data_config = config['Data']
    net_config = config['Net']

    target_size = data_config["target_size"]

    model = load_model(**net_config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # To device
    model = model.to(device)

    if torch.cuda.is_available():
        model = nn.DataParallel(model)

    model_path = "/media/2tb/output_models/headpose_resnet/model_epoch_78_3.985460854345752.pth"
    # model_path = "/media/2tb/output_models/headpose_fsanet/model_epoch_228_5.439456623458148.pth"
    # model_path = "/home/linhnv/projects/fsanet_pytorch/model_train/FSA_MSE/model_epoch_77_5.65551967774668.pth"
    param = torch.load(model_path, map_location='cpu')
    model.load_state_dict(param)
    model.to(device)

    raw_path = '/home/linhnv/projects/HeadPoseEstimation-WHENet/Sample/AILab_imgs'
    root_path = Path(raw_path)

    with open(root_path / 'bbox.txt', 'r') as f:
        lines = f.readlines()

    for l in lines:
        print(f"[INFO] Read line: {l}")
        filename, bboxes =l.split(',')[0], l.split(',')[1:]
        bboxes = [bbox.split(' ')[1:] for bbox in bboxes]
        # print(bboxes)
        bboxes = [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in bboxes]
        # print(bboxes)
        
        crop_and_pred(raw_path+'/'+filename,bboxes, model, device, target_size, scale=0.5)