from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

from dataset import get_train_test_loaders


class Net(nn.Module):

    # building the neural net model
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 6, 3)
        self.conv3 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 48)
        self.fc3 = nn.Linear(48, 25)

    # the forward pass / how data is passed through the model
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    net = Net().float()
    criterion = nn.CrossEntropyLoss()   # evaluation metric

    # gradient step optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # load dataset & define how many times we train
    trainloader, _ = get_train_test_loaders()
    NUM_EPOCHS = 12

    # for each time we loop through dataset, train & take 1 gradient step
    for epoch in range(NUM_EPOCHS):
        train(net, criterion, optimizer, trainloader, epoch)
        scheduler.step()
    torch.save(net.state_dict(), "classifier_model/checkpoint.pth")


def train(net, criterion, optimizer, trainloader, epoch):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        
        # get data, reset gradients
        inputs = Variable(data['image'].float())
        labels = Variable(data['label'].long())
        optimizer.zero_grad()

        # forward pass, backward pass, optimize step
        outputs = net(inputs)
        loss = criterion(outputs, labels[:, 0])
        loss.backward()
        optimizer.step()

        # print loss value for every 100 samples seen
        running_loss += loss.item()
        if i % 100 == 0:
            print(f"[{epoch:d}, {i:4d}] loss: {running_loss / (i + 1):.6f}")


if __name__ == '__main__':
    main()