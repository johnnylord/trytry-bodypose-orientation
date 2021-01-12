import os
import os.path as osp

import numpy as np
import pandas as pd
from PIL import Image

import torch


__all__ = [ "OrientDataset" ]

class OrientDataset:

    def __init__(self, root, img_transform=None, pose_transform=None, size=(256, 128)):
        self.root = root
        self.img_transform = img_transform
        self.pose_transform = pose_transform
        self.size = size
        # Prepare files
        img_dir = osp.join(root, 'img')
        pose_dir = osp.join(root, 'pose')
        ifiles = sorted([ osp.join(img_dir, f) for f in os.listdir(img_dir) ])
        pfiles = sorted([ osp.join(pose_dir, f) for f in os.listdir(pose_dir) ])
        # Extract metadata
        metadata = pd.read_csv(osp.join(root, 'orient.csv'))
        labels = metadata['label']
        indices = [ f.split(".")[0] for f in metadata['filename'] ]
        # Extract valid data samples
        valid_ifiles = []
        valid_pfiles = []
        valid_labels = []
        for ifile, pfile in zip(ifiles, pfiles):
            iid = osp.basename(ifile).split(".")[0]
            pid = osp.basename(pfile).split(".")[0]
            if iid not in indices:
                continue
            target_idx = indices.index(iid)
            target_label = labels[target_idx]
            valid_ifiles.append(ifile)
            valid_pfiles.append(pfile)
            valid_labels.append(target_label)
        self.samples = list(zip(valid_ifiles, valid_pfiles, valid_labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ifile, pfile, label = self.samples[idx]

        img = Image.open(ifile)
        old_size = img.size
        new_size = self.size[::-1]

        img = img.resize(self.size[::-1])
        width, height = img.size
        if self.img_transform is not None:
            img = self.img_transform(img)

        pose = np.load(pfile)
        x_scale, y_scale = np.array(old_size)/np.array(new_size)
        pose[:, 0] *= x_scale
        pose[:, 1] *= y_scale
        pose[:, 0] /= width
        pose[:, 1] /= height
        pose = pose.astype(np.float32)
        if self.pose_transform is not None:
            pose = self.pose_transform(pose)

        return img, pose, label
