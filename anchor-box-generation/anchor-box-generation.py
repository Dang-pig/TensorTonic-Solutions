import numpy as np
import itertools as it

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    stride = image_size / feature_size
    sr = list(it.product(scales, aspect_ratios))
    sr = [[x, y] for x, y in sr]
    sr = np.asarray(sr)
    x = (np.arange(feature_size) + 0.5) * stride
    y = x.copy()
    c = np.zeros((y.shape[0], x.shape[0], sr.shape[0], 4))
    x = x.reshape(1, -1, 1)
    y = y.reshape(-1, 1, 1)
    w = sr[:, 0] * np.sqrt(sr[:, 1])
    h = sr[:, 0] / np.sqrt(sr[:, 1])
    w = w.reshape(1, 1, -1)
    h = h.reshape(1, 1, -1)
    c[:, :, :, 0] = x
    c[:, :, :, 1] = y
    c[:, :, :, 2] = c[:, :, :, 0] + w / 2
    c[:, :, :, 3] = c[:, :, :, 1] + h / 2
    c[:, :, :, 0] -= w / 2
    c[:, :, :, 1] -= h / 2
    c = c.reshape(-1, 4)
    return c.tolist()