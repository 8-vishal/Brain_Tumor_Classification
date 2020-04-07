import torchvision
import torch.nn as nn
import torch.optim as optim
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


def TRAINING(save_model, train_loader, EPOCH):
    model = CNN_2D()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(EPOCH):
        running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        #print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
        running_loss = 0.0
    print('Finished Training')
    if save_model:
        path = './brain_tumor_CNN.pth'
        return torch.save(CNN_2D().state_dict(), path)


def TESTING(trained_model):
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    model = CNN_2D()
    model.load_state_dict(torch.load(trained_model))
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return print('Accuracy of the network on test images: %d %%' % (100 * correct / total))


train_loader, test_loader = DATA_LOADER()
#TRAINING(True, train_loader, 100)
TESTING('brain_tumor_CNN.pth')