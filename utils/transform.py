import numpy as np

def scale_bboxes_coord(bboxes, old_resolution, new_resolution):
    ratio = np.array(new_resolution) / np.array(old_resolution)
    x_scale = ratio[1]
    y_scale = ratio[0]
    bboxes[:, 0] *= x_scale
    bboxes[:, 1] *= y_scale
    bboxes[:, 2] *= x_scale
    bboxes[:, 3] *= y_scale
    return bboxes

def scale_keypoints_coord(keypoints, old_resolution, new_resolution):
    ratio = np.array(new_resolution) / np.array(old_resolution)
    x_scale = ratio[1]
    y_scale = ratio[0]
    keypoints[:, 0] *= x_scale
    keypoints[:, 1] *= y_scale
    return keypoints

