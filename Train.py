from Brain_Tumor.model import CNN_2D, DATA_LOADER
import torch.nn as nn
import torch.optim as optim
import torch


model = CNN_2D()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_loader, test_loader = DATA_LOADER()

for epoch in range(15):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()    # print every 2000 mini-batches
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
        running_loss = 0.0

print('Finished Training')
PATH = './Brain_tumor_classifier.pth'
torch.save(CNN_2D().state_dict(), PATH)