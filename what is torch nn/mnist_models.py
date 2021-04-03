import math

import torch
import torch.nn.functional as F
from torch import nn

"""
Models functions
"""


def create_mnist_model_sequential():
    model = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        Lambda(lambda x: x.view(x.size(0), -1)),  # or lambda outprocess
    )
    return model


class MnistLogistic(nn.Module):
    def __init__(self):
        # In the init method of the class we can define parameters, layers or other components of the model.
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        xb = xb.view(-1, 28 * 28)
        return xb @ self.weights + self.bias


class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # input image size: 28*28
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        # image size 14
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        # image size 7
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)
        # image size 4

    def forward(self, xb):
        # In the forward method we define the model operations and their order
        # First reshape the input
        # add channel dimension (grey scale = 1 channel) and divide the 784 pixels (28x28)
        xb = xb.view(-1, 1, 28, 28)
        # Use conv layers followed by activation function relu
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        # Perform pooling operation
        xb = F.avg_pool2d(xb, 4)
        # Output:  maintain batch dimension, remove the last two dimensions (they are size 1 and 1)
        # view is like reshape. To remove dimensions of size 1 it can be possible to use squeeze method.
        # xb.squeeze()
        xb = xb.view(-1, xb.size(1))
        return xb


"""
Auxiliary
"""


class Lambda(nn.Module):
    """
    Lambda layer. Apply a custom function and use it when building a model.
    """

    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
