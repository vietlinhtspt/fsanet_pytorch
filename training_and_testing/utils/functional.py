from pickle import TRUE
import scipy.io as sio
import numpy as np
import torch
import math

def quat2euler(w, x, y, z):
    roll = math.atan2(2*(w*x + y*z), 1-2*(x*x + y*y))
    pitch = math.asin(2*(w*y - z*x))
    yaw = math.atan2(2*(w*z + y*x), 1-2*(z*z + y*y))

    roll = roll*180 / math.pi
    pitch = pitch*180 / math.pi
    yaw = yaw*180 / math.pi

    return [yaw, pitch, roll]

def get_pt_ypr_from_mat(mat_path, pt3d=False):
    # Get 2D landmarks
    mat = sio.loadmat(mat_path)
    if pt3d:
        pt = mat['pt3d_68']
    else:
        pt = mat['pt2d']

    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll]
    pose = pre_pose_params[:3]

    x_min = min(pt[0,:])
    y_min = min(pt[1,:])
    x_max = max(pt[0,:])
    y_max = max(pt[1,:])

    
    pitch = pose[0] * 180 / np.pi
    yaw = pose[1] * 180 / np.pi
    roll = pose[2] * 180 / np.pi

    return (x_min, y_min, x_max, y_max), (yaw, pitch, roll)

def l2_norm(input__, axis=1):
    norm = torch.norm(input__, 2, axis, True)
    return torch.div(input__, norm)

#yaw, pitch, roll -> W, Z, Y, X
def euler2quat(yaw, pitch, roll):
    roll = roll * math.pi / 180
    pitch = pitch * math.pi / 180
    yaw = yaw * math.pi / 180

    x = math.sin(pitch / 2) * math.sin(yaw / 2) * math.cos(roll / 2) + math.cos(pitch / 2) * math.cos(yaw / 2) * math.sin(roll / 2)
    y = math.sin(pitch / 2) * math.cos(yaw / 2) * math.cos(roll / 2) + math.cos(pitch / 2) * math.sin(yaw / 2) * math.sin(roll / 2)
    z = math.cos(pitch / 2) * math.sin(yaw / 2) * math.cos(roll / 2) - math.sin(pitch / 2) * math.cos(yaw / 2) * math.sin(roll / 2)
    w = math.cos(pitch / 2) * math.cos(yaw / 2) * math.cos(roll / 2) - math.sin(pitch / 2) * math.sin(yaw / 2) * math.sin(roll / 2)
    return [w, x, y, z]