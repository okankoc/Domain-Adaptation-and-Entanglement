# Probabilistic Representation Networks that double the last layer output of each model found in conv.py
import copy
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions


# Modified from DANN script
class ProbConvNet2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = 200
        self.name = "conv2_prob"

        self.net = nn.Sequential()
        self.net.add_module("conv1", nn.LazyConv2d(64, kernel_size=5))
        self.net.add_module("bn1", nn.BatchNorm2d(64))
        self.net.add_module("pool1", nn.MaxPool2d(2))
        self.net.add_module("relu1", nn.ReLU(True))
        self.net.add_module("conv2", nn.Conv2d(64, 50, kernel_size=5))
        self.net.add_module("bn2", nn.BatchNorm2d(50))
        self.net.add_module("drop1", nn.Dropout())
        self.net.add_module("pool2", nn.MaxPool2d(2))
        self.net.add_module("relu2", nn.ReLU(True))
        self.net.add_module("flatten", nn.Flatten())
        self.net.add_module("fc1", nn.LazyLinear(100))
        self.net.add_module("bn3", nn.BatchNorm1d(100))
        self.net.add_module("relu3", nn.ReLU(True))
        self.net.add_module("drop2", nn.Dropout())
        self.net.add_module("fc2", nn.Linear(100, self.num_features))
        self.net.add_module("bn4", nn.BatchNorm1d(self.num_features))
        self.net.add_module("last_features", nn.ReLU(True))
        self.last_layer = nn.Linear(int(self.num_features / 2), num_classes)

    def forward(self, input_data):
        features = self.net(input_data)
        mean = features[:, : int(self.num_features / 2)]
        std = features[:, int(self.num_features / 2) :]
        normal_dist = torch.distributions.normal.Normal(mean, F.softplus(std))
        feat_dist = torch.distributions.Independent(base_distribution=normal_dist, reinterpreted_batch_ndims=1)
        # Reparameterized sample
        sample = feat_dist.rsample()
        output = self.last_layer(sample)
        return output

    def forward_distr(self, input_data):
        features = self.net(input_data)
        mean = features[:, : int(self.num_features / 2)]
        std = features[:, int(self.num_features / 2) :]
        normal_dist = torch.distributions.normal.Normal(mean, F.softplus(std))
        feat_dist = torch.distributions.Independent(base_distribution=normal_dist, reinterpreted_batch_ndims=1)
        # Reparameterized sample
        sample = feat_dist.rsample()
        output = self.last_layer(sample)
        return mean, std, sample, output, feat_dist

    def copy(self, device):
        new_model = ProbConvNet2(self.num_classes).to(device)
        new_model.load_state_dict(self.state_dict())
        new_model.save_params()
        return new_model

    @torch.no_grad()
    def save_params(self):
        self.state = copy.deepcopy(self.state_dict())

    @torch.no_grad()
    def restore_params(self):
        self.load_state_dict(self.state)
        return dict(self.named_parameters())


class ProbConvNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = 256
        self.name = "conv1_prob"
        self.conv = nn.Sequential()
        self.conv.add_module("conv1", nn.LazyConv2d(32, kernel_size=3, stride=1, padding="same"))
        self.conv.add_module("relu1", nn.ReLU(True))
        self.conv.add_module("conv2", nn.Conv2d(32, 32, kernel_size=3, stride=1, padding="same"))
        self.conv.add_module("relu2", nn.ReLU(True))
        self.conv.add_module("pool1", nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv.add_module("conv3", nn.Conv2d(32, 64, kernel_size=3, stride=1, padding="same"))
        self.conv.add_module("relu3", nn.ReLU(True))
        self.conv.add_module("conv4", nn.Conv2d(64, 64, kernel_size=3, stride=1, padding="same"))
        self.conv.add_module("relu4", nn.ReLU(True))
        self.conv.add_module("pool2", nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv.add_module("conv5", nn.Conv2d(64, 128, kernel_size=3, stride=1, padding="same"))
        self.conv.add_module("relu5", nn.ReLU(True))
        self.conv.add_module("conv6", nn.Conv2d(128, 128, kernel_size=3, stride=1, padding="same"))
        self.conv.add_module("relu6", nn.ReLU(True))
        self.conv.add_module("pool3", nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv.add_module("flatten", nn.Flatten())
        self.conv.add_module("fc1", nn.LazyLinear(self.num_features))
        self.conv.add_module("last_features", nn.ReLU())
        self.last_layer = nn.Linear(int(self.num_features / 2), num_classes)

    def copy(self, device):
        new_model = ProbConvNet(self.num_classes).to(device)
        new_model.load_state_dict(self.state_dict())
        new_model.save_params()
        return new_model

    def forward(self, input_data):
        features = self.conv(input_data)
        mean = features[:, : int(self.num_features / 2)]
        std = features[:, int(self.num_features / 2) :]
        normal_dist = torch.distributions.normal.Normal(mean, F.softplus(std))
        feat_dist = torch.distributions.Independent(base_distribution=normal_dist, reinterpreted_batch_ndims=1)
        # Reparameterized sample
        sample = feat_dist.rsample()
        output = self.last_layer(sample)
        return output

    def forward_distr(self, input_data):
        features = self.conv(input_data)
        mean = features[:, : int(self.num_features / 2)]
        std = features[:, int(self.num_features / 2) :]
        normal_dist = torch.distributions.normal.Normal(mean, F.softplus(std))
        feat_dist = torch.distributions.Independent(base_distribution=normal_dist, reinterpreted_batch_ndims=1)
        # Reparameterized sample
        sample = feat_dist.rsample()
        output = self.last_layer(sample)
        return mean, std, sample, output, feat_dist

    @torch.no_grad()
    def save_params(self):
        self.state = copy.deepcopy(self.state_dict())

    @torch.no_grad()
    def restore_params(self):
        self.load_state_dict(self.state)
        return dict(self.named_parameters())


class ProbLeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = 168
        self.name = "lenet_prob"
        self.net = nn.Sequential()
        self.net.add_module("conv1", nn.LazyConv2d(6, kernel_size=5, padding=2))
        self.net.add_module("relu1", nn.ReLU())
        self.net.add_module("avg_pool1", nn.AvgPool2d(kernel_size=2, stride=2))
        self.net.add_module("conv2", nn.LazyConv2d(16, kernel_size=5))
        self.net.add_module("relu2", nn.ReLU())
        self.net.add_module("avg_pool2", nn.AvgPool2d(kernel_size=2, stride=2))
        self.net.add_module("flatten", nn.Flatten())
        self.net.add_module("fc1", nn.LazyLinear(120))
        self.net.add_module("relu3", nn.ReLU())
        self.net.add_module("fc2", nn.LazyLinear(self.num_features))
        self.net.add_module("last_features", nn.ReLU())
        self.last_layer = nn.Linear(int(self.num_features / 2), num_classes)

    def forward(self, input_data):
        features = self.net(input_data)
        mean = features[:, : int(self.num_features / 2)]
        std = features[:, int(self.num_features / 2) :]
        normal_dist = torch.distributions.normal.Normal(mean, F.softplus(std))
        feat_dist = torch.distributions.Independent(base_distribution=normal_dist, reinterpreted_batch_ndims=1)
        # Reparameterized sample
        sample = feat_dist.rsample()
        output = self.last_layer(sample)
        return output

    def forward_distr(self, input_data):
        features = self.net(input_data)
        mean = features[:, : int(self.num_features / 2)]
        std = features[:, int(self.num_features / 2) :]
        normal_dist = torch.distributions.normal.Normal(mean, F.softplus(std))
        feat_dist = torch.distributions.Independent(base_distribution=normal_dist, reinterpreted_batch_ndims=1)
        # Reparameterized sample
        sample = feat_dist.rsample()
        output = self.last_layer(sample)
        return mean, std, sample, output, feat_dist

    def copy(self, device):
        new_model = ProbLeNet(self.num_classes).to(device)
        new_model.load_state_dict(self.state_dict())
        new_model.save_params()
        return new_model

    @torch.no_grad()
    def save_params(self):
        self.state = copy.deepcopy(self.state_dict())


# Define a small CNN model
class ProbSmallCNN(nn.Module):
    def __init__(self, num_classes):
        super(ProbSmallCNN, self).__init__()
        self.num_classes = num_classes
        self.num_features = 256
        self.name = "small_cnn_prob"
        self.net = nn.Sequential()
        self.net.add_module("conv1", nn.LazyConv2d(32, kernel_size=3, stride=1, padding=1))
        self.net.add_module("relu1", nn.ReLU())
        self.net.add_module("max_pool1", nn.MaxPool2d(kernel_size=2))
        self.net.add_module("conv2", nn.LazyConv2d(64, kernel_size=3, stride=1, padding=1))
        self.net.add_module("relu2", nn.ReLU())
        self.net.add_module("max_pool2", nn.MaxPool2d(kernel_size=2))
        self.net.add_module("flatten", nn.Flatten())
        self.net.add_module("fc1", nn.LazyLinear(self.num_features))
        self.net.add_module("last_features", nn.ReLU())
        self.last_layer = nn.Linear(int(self.num_features / 2), num_classes)

    def forward(self, input_data):
        features = self.net(input_data)
        mean = features[:, : int(self.num_features / 2)]
        std = features[:, int(self.num_features / 2) :]
        normal_dist = torch.distributions.normal.Normal(mean, F.softplus(std))
        feat_dist = torch.distributions.Independent(base_distribution=normal_dist, reinterpreted_batch_ndims=1)
        # Reparameterized sample
        sample = feat_dist.rsample()
        output = self.last_layer(sample)
        return output

    def forward_distr(self, input_data):
        features = self.net(input_data)
        mean = features[:, : int(self.num_features / 2)]
        std = features[:, int(self.num_features / 2) :]
        normal_dist = torch.distributions.normal.Normal(mean, F.softplus(std))
        feat_dist = torch.distributions.Independent(base_distribution=normal_dist, reinterpreted_batch_ndims=1)
        # Reparameterized sample
        sample = feat_dist.rsample()
        output = self.last_layer(sample)
        return mean, std, sample, output, feat_dist

    def copy(self, device):
        new_model = ProbSmallCNN(self.num_classes).to(device)
        new_model.load_state_dict(self.state_dict())
        return new_model
