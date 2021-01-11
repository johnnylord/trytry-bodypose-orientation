import os
import os.path as osp

import torch
import torch.nn as nn


class OrientAgent:

    def __init__(self, config):
        self.config = config

        # Determine deeplearning environment
        device = config['train']['device'] if torch.cuda.is_available() else "cpu"
        self.device = device

        # Dataset

        # Model

        # Optimizer

        # Loss function

        # Resume training

    def train(self):
        pass

    def train_one_epoch(self):
        pass

    def validate(self):
        pass

    def finalize(self):
        pass

    def _save_checkpoint(self):
        pass
