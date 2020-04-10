import torch
from Brain_Tumor.model import DATA_LOADER, CNN_2D


train_loader, test_loader = DATA_LOADER()
dataiter = iter(test_loader)
images, labels = dataiter.next()
model = CNN_2D()
model.load_state_dict(torch.load('./Brain_tumor_classifier.pth'))
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on test images: %d %%' % (100 * correct / total))

