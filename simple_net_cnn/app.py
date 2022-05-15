import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from simple_net_cnn.cnn import Net
from simple_net_cnn.dataset import get_data
from simple_net_cnn.train import train_net
from simple_net_cnn.testing import test_net


logger = logging.getLogger(__name__)

model_path = Path('simple_net_cnn/models/cat_dogs.pth')


def image_show(dataset: DataLoader, numbers: int = 5):
    """Функция показывает первые 5 (по умолчанию) картинок датасета"""

    sorts = ('cat', 'dog')
    dataiter = iter(dataset)
    images, labels = dataiter.next()
    fig, axes = plt.subplots(figsize=(10, 4), ncols=numbers)
    for number in range(numbers):
        ax = axes[number]
        ax.imshow(images[number].permute(1, 2, 0))
        ax.title.set_text(' '.join('%5s' % sorts[labels[number]]))
    plt.show()


def run():
    logger.debug('start app.run')
    train, _ = get_data()
    net = train_net(dataset=train, epochs=10)
    torch.save(net.state_dict(), str(model_path))


def test():
    logger.debug('start testing')
    _, test = get_data()
    logger.debug(test.batch_size)
    correct, total = test_net(net_path=model_path, dataset=test)
    message = 'Accuracy of the network on the {0} test images: {1}%'.format(
        len(test),
        100 * correct / total,
    )
    logger.debug(message)
