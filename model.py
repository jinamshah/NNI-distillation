# Importing required libraries
from __future__ import print_function
import argparse
import nni
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchsummary import summary
import numpy as np


# Creating arguments parser
parser = argparse.ArgumentParser(description='ResNet light CIFAR-10 Example')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

# Setting the random seed
torch.manual_seed(args.seed)

# Using GPU, if available
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Device selected for training and testing
device = torch.device("cuda" if args.cuda else "cpu")
print("Device used: ", device, "\n")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# Defining parameters to be tuned
params = {
    'dropout_rate': 0.0,
    'lr': 0.001,
    'momentum': 0,
    "batch_size": 64
}

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True


# Loading the training and testing data in dataloaders
train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=params['batch_size'], shuffle=True, drop_last=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=params['batch_size'], shuffle=False, drop_last=True, **kwargs)

# Saving the test data for use with tvm later
data_numpy = next(iter(test_loader))[0].numpy()
np.savez("test_data", data=data_numpy)

# Creating a ResNet block
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)

# Creating a ResNet architecture
class ResNet(nn.Module):
    def __init__(self, in_channels, resblock, repeat, outputs=10):
        super().__init__()
        self.dropout = nn.Dropout(params['dropout_rate'])
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Specifying custom filter numbers
        filters = [64, 64, 256]

        self.layer1 = nn.Sequential()
        self.layer1.add_module('conv2_1', resblock(filters[0], filters[1], downsample=False))
        for i in range(1, repeat[0]):
            self.layer1.add_module('conv2_%d' % (i + 1,), resblock(filters[1], filters[1], downsample=False))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('conv3_1', resblock(filters[1], filters[2], downsample=True))
        for i in range(1, repeat[1]):
            self.layer2.add_module('conv3_%d' % (
                i + 1,), resblock(filters[2], filters[2], downsample=False))

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(filters[2], outputs)

    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.gap(input)
        input = torch.flatten(input, start_dim=1)
        input = self.dropout(input)
        input = self.fc(input)
        return input



# Specifying the loss function
loss_fn = nn.CrossEntropyLoss()


# Model training function
def train(model, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()


# Model testing function
def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    size = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    correct /= size
    return correct

# Defining a class for distillation to override the nni module.
# This will help us calculate the KL Divergence
class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = torch.nn.functional.log_softmax(y_s / self.T, dim=1)
        p_t = torch.nn.functional.softmax(y_t / self.T, dim=1)
        loss = torch.nn.functional.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

# Function for distillation of a student model using a teacher model
def fine_tune(models, optimizer,kd_temperature):
    model_s = models[0].train()
    model_t = models[-1].eval()
    # cri_cls = criterion
    cri_kd = DistillKL(kd_temperature)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        y_s = model_s(data)
        y_t = model_t(data)

        loss_cri = torch.nn.functional.cross_entropy(y_s, target)
        loss_kd = cri_kd(y_s, y_t)
        # total loss
        loss = loss_cri + loss_kd
        loss.backward()
        optimizer.step()

