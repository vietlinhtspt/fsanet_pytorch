import numpy as np

def preprocess(img):
    img = img / 256.
    img = (img - np.asarray([0.485, 0.456, 0.406])) / np.asarray([0.229, 0.224, 0.225])
    return img

def change_bbox(bbox, scale=1, use_forehead=True):
    x_min, y_min, x_max, y_max = bbox
    if use_forehead:
        k = scale
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
    else:
        w, h = x_max-x_min, y_max-y_min
        h_w = max(h, w) * scale
        x_min -= (h_w-w)//2
        y_min -= (h_w-h)//2
        x_max = x_min + h_w
        y_max = y_min + h_w
    return (int(x_min), int(y_min), int(x_max), int(y_max))