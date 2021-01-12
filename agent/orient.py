import os
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tensorboardX import SummaryWriter

from data.dataset import OrientDataset
from model.orient import OrientNet

class OrientAgent:

    def __init__(self, config):
        self.config = config

        # Determine deeplearning environment
        device = config['train']['device'] if torch.cuda.is_available() else "cpu"
        self.device = device

        # Dataset
        img_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ])
        pose_transform = T.Compose([
            T.ToTensor(),
            ])
        train_dataset = OrientDataset(root=config['dataset']['train']['root'],
                                    img_transform=img_transform,
                                    pose_transform=pose_transform,
                                    size=config['dataset']['size'])
        valid_dataset = OrientDataset(root=config['dataset']['valid']['root'],
                                    img_transform=img_transform,
                                    pose_transform=pose_transform,
                                    size=config['dataset']['size'])
        self.train_loader = DataLoader(dataset=train_dataset,
                                    batch_size=config['dataloader']['batch_size'],
                                    num_workers=config['dataloader']['num_workers'],
                                    shuffle=True)
        self.valid_loader = DataLoader(dataset=valid_dataset,
                                    batch_size=config['dataloader']['batch_size'],
                                    num_workers=config['dataloader']['num_workers'],
                                    shuffle=False)

        # Model
        self.model = OrientNet(n_keypoints=config['model']['n_keypoints'],
                                n_orients=config['model']['n_orients'])
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=config['optimizer']['lr'])

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Tensorboard writer
        self.writer = SummaryWriter(logdir=osp.join(config['train']['logdir'],
                                                    config['train']['exp_name']))
        self.current_acc = 0.0
        self.current_epoch = -1
        self.remain_patients = config['train']['n_patients']

        # Resume training
        if config['train']['resume']:
            checkpoint_dir = osp.join(config['train']['logdir'],
                                    "{}_checkpoint".format(config['train']['exp_name']))
            checkpoint = torch.load(osp.join(checkpoint_dir, 'best.pth'))
            # Load pretrained state
            self.model = checkpoint['model']
            self.optimizer = checkpoint['optimizer']
            self.current_acc = checkpoint['current_acc']
            self.current_epoch = checkpoint['current_epoch']
            print(f"Resume training at epoch '{self.current_acc}'")

    def train(self):
        for epoch in range(self.current_epoch+1, self.config['train']['n_epochs']):
            self.current_epoch = epoch
            self.train_one_epoch()
            self.validate()
            if self.remain_patients < 0:
                print(f"Early stop at epoch '{self.current_epoch}'")
                break

    def train_one_epoch(self):
        running_loss = 0
        running_correct = 0

        self.model.train()
        for batch_idx, (imgs, poses, labels) in enumerate(self.train_loader):
            batch_size = imgs.size(0)
            # Move to proper device
            imgs = imgs.to(self.device)
            poses = poses.to(self.device)
            labels = labels.to(self.device)
            # Model forward & backward
            self.optimizer.zero_grad()
            outputs = self.model(poses.view(batch_size, -1))
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            # Predicts
            preds = torch.max(outputs, 1)[1]
            corrects = preds.eq(labels).sum()
            running_correct += corrects.item()
            running_loss += loss.item()*batch_size
            # Logging
            if batch_idx % self.config['train']['log_interval'] == 0:
                print("Epoch {}:{}, Train Loss {:.3f}".format(
                    self.current_epoch, self.config['train']['n_epochs'],
                    loss.item()))

        epoch_loss = running_loss / (len(self.train_loader.dataset)*batch_size)
        epoch_acc = running_correct / (len(self.train_loader.dataset)*batch_size)
        print("Epoch {}:{}, Train Loss {:.3f}, Train Acc {:.3f}".format(
            self.current_epoch, self.config['train']['n_epochs'],
            epoch_loss, epoch_acc))
        self.writer.add_scalar("Train Loss", epoch_loss, self.current_epoch)
        self.writer.add_scalar("Train Acc", epoch_acc, self.current_epoch)

    def validate(self):
        running_loss = 0
        running_correct = 0

        self.model.eval()
        for batch_idx, (imgs, poses, labels) in enumerate(self.valid_loader):
            batch_size = imgs.size(0)
            # Move to proper device
            imgs = imgs.to(self.device)
            poses = poses.to(self.device)
            labels = labels.to(self.device)
            # Model forward & backward
            outputs = self.model(poses.view(batch_size, -1))
            loss = self.criterion(outputs, labels)
            # Predicts
            preds = torch.max(outputs, 1)[1]
            corrects = preds.eq(labels).sum()
            running_correct += corrects.item()
            running_loss += loss.item()*batch_size

        epoch_loss = running_loss / (len(self.valid_loader.dataset)*batch_size)
        epoch_acc = running_correct / (len(self.valid_loader.dataset)*batch_size)
        print("Epoch {}:{}, Valid Loss {:.3f}, Valid Acc {:.3f}".format(
            self.current_epoch, self.config['train']['n_epochs'],
            epoch_loss, epoch_acc))
        self.writer.add_scalar("Valid Loss", epoch_loss, self.current_epoch)
        self.writer.add_scalar("Valid Acc", epoch_acc, self.current_epoch)

        if epoch_acc > self.current_acc:
            self.current_acc = epoch_acc
            self._save_checkpoint()

    def finalize(self):
        pass

    def _save_checkpoint(self):
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_acc': self.current_acc,
            'current_epoch': self.current_epoch,
            }
        checkpoint_dir = osp.join(self.config['train']['logdir'],
                                "{}_checkpoint".format(self.config['train']['exp_name']))
        if not osp.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        path = osp.join(checkpoint_dir, 'best.pth')
        torch.save(checkpoint, path)
        print("Save checkpoint to '{}'".format(path))
