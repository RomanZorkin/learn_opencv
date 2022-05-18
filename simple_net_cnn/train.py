import logging

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

from simple_net_cnn.cnn import Net

logger = logging.getLogger(__name__)

net = Net()


def train_net(dataset: DataLoader, net: Net = net, epochs: int = 10,) -> Net:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    losses = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for num, data in enumerate(dataset):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # log statistics
            losses.append(loss)
            running_loss += loss.item()
            if num % 10 == 9:  # log every 2000 mini-batches
                message = '[{0}, {1}] loss: {2:.10f}'.format(
                    epoch + 1,
                    num + 1,
                    running_loss / 200,
                )
                logger.info(message)
    
    # Здесь возможна ошибка https://github.com/pytorch/pytorch/issues/44023
    # Для решения перейти по ссылке в traceback и в методе __array__
    # заменить return self.numpy() на return self.detach().numpy()
    plt.plot(losses, label='Training loss')
    plt.show()
    logger.info('Finish training')

    return net
