import torch
import torch.nn as nn


class OrientNet(nn.Module):
    """Orientation Network to predict the orientation of a body pose

    The keypoints of bodypose is obtained from openpose pose-estimation model
    with 25 keypoints. Each keypoint consists of three values (x, y, conf).

    The orientation of a bodypose is divided into 8 category
    """
    def __init__(self, n_keypoints=25, n_orients=8):
        super().__init__()
        self.n_keypoints = n_keypoints
        self.n_orients = n_orients
        self.feature = nn.Sequential(
                nn.Linear(n_keypoints*3, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 32),
                nn.ReLU(inplace=True))
        self.fc = nn.Sequential(nn.Linear(32, n_orients))

    def forward(self, x):
        feature = self.feature(x)
        output = self.fc(feature)
        return output


if __name__ == "__main__":
    from torchsummary import summary

    x = torch.rand(3, 25*3)
    model = OrientNet()
    y = model(x)
    print(x.shape, y.shape)
