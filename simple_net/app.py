import logging
from importlib_metadata import requires

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


logger = logging.getLogger(__name__)


def simple_gradient():
    logger.debug('start simple_gradient')
    x = Variable(torch.ones(2, 2) * 2, requires_grad=True)
    z = 2 * (x * x) + 5 * x
    z.backward(torch.ones(2, 2))
    logger.debug(f'{x.grad}, z = {z}, x = {x}')
    y = torch.ones(2, 3) * 3
    logger.debug(y)


def create_nn(batch_size: int, learning_rate:float, epochs: int, log_interval: int):
    dataset_train = datasets.MNIST(
        'simple_net/data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081)),
        ]),
    )
    dataset_test = datasets.MNIST(
        'simple_net/data',
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081)),
        ]),
    )

    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    test_loader =DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True)


    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(28 * 28, 200)
            self.fc2 = nn.Linear(200, 200)
            self.fc3 = nn.Linear(200, 10)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return F.log_softmax(x)

    net = Net()
    logger.debug(net)

    # create a stochastic gradient descent optimizer
    optimazer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    # create a loss function
    criterion = nn.NLLLoss()

    # run the main training loop
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
            data = data.view(-1, 28 * 28)
            optimazer.zero_grad()
            net_out = net(data)
            loss = criterion(net_out, target)
            loss.backward()
            optimazer.step()
            logger.debug(loss)
            if batch_idx % log_interval == 0:
                message = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item(),
                )
                logger.info(message)
    # run test loop
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        data = data.view(-1, 28 * 28)
        net_out = net(data)
        # sum up batch loss
        test_loss += criterion(net_out, target).item()
        pred = net_out.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).sum()

    test_loss /= len(test_loader.dataset)
    message = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss,
        correct,
        len(test_loader.dataset),
        100. * correct / len(test_loader.dataset),
    )
    logger.info(message)
