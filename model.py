import torchvision
import torch.nn as nn
from torchvision import transforms
import torch.utils.data as data
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CNN_2D(nn.Module):
    def __init__(self):
        super(CNN_2D, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 264, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2))
        self.layer4 = nn.Sequential(
            nn.Linear(14 * 14 * 264, 1000, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.2))
        self.layer5 = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Dropout(p=0.2))
        self.layer6 = nn.Sequential(
            nn.Linear(500, 2))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        return out


def DATA_LOADER():
    train_folder = "brain_tumor_dataset/train/"
    test_folder = "brain_tumor_dataset/test/"
    BATCH_SIZE = 10
    Transforming = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation((30, 120)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.4, 0.5], std=[0.22, 0.24, 0.22])])
    train_data = torchvision.datasets.ImageFolder(root=train_folder, transform=Transforming)
    test_data = torchvision.datasets.ImageFolder(root=test_folder, transform=Transforming)
    train_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    return train_loader, test_loader

