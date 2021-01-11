import numpy as np
from scipy.optimize import linear_sum_assignment
from utils.cost import compute_iou_dist

def match_bbox_keypoint(bboxes, all_keypoints):
    # Group bboxes & keypoints together
    kbboxes = []
    for keypoints in all_keypoints:
        valids = keypoints[(keypoints[:, 0]*keypoints[:, 1])>0, :]
        xmin = np.min(valids[:, 0])
        xmax = np.max(valids[:, 0])
        ymin = np.min(valids[:, 1])
        ymax = np.max(valids[:, 1])
        kbboxes.append([xmin, ymin, xmax, ymax])
    kbboxes = np.array(kbboxes)

    # Greedy matching
    if len(bboxes) > 0 and len(kbboxes) > 0:
        shape = np.array(list(bboxes.shape), dtype=np.int)
        shape[-1] = shape[-1] + 25*3
        shape = tuple(shape.tolist())
        bodyposes = np.zeros(shape)
        bodyposes[:, :5] = bboxes
        iou_matrix = compute_iou_dist(bboxes[:, :4], kbboxes)
        bindices, kindices = linear_sum_assignment(iou_matrix)
        for bidx, kidx in zip(bindices, kindices):
            poses = all_keypoints[kidx].reshape(-1)
            bodyposes[bidx, 5:] = poses
    else:
        bodyposes = []

    return bodyposes
