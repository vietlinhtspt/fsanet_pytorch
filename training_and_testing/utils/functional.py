import scipy.io as sio
import numpy as np

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
