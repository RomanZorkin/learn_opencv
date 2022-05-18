import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from simple_net_cnn import config
from simple_net_cnn.convertor import change_type
from simple_net_cnn.dataset import get_data
from simple_net_cnn.testing import test_net
from simple_net_cnn.train import train_net

logger = logging.getLogger(__name__)

model_path = config.MODEL_PATH
whole_model_path = config.WHOLE_MODEL_PATH

def image_show(dataset: DataLoader, numbers: int = 5):
    """Функция показывает первые 5 (по умолчанию) картинок датасета."""

    # извлекаем список объектов для распознования предусмотренных моделью
    sorts = dataset.dataset.classes

    dataiter = iter(dataset)
    images, labels = dataiter.next()
    logger.debug(labels)
    fig, axes = plt.subplots(figsize=(10, 4), ncols=numbers)
    for number in range(numbers):
        ax = axes[number]
        ax.imshow(images[number].permute(1, 2, 0))
        ax.title.set_text(' '.join('%5s' % sorts[labels[number]]))
    plt.show()


def run():
    logger.debug('start app.run')
    train, _, classes, sizes = get_data()
    logger.debug(f'Train dataset: {train}')
    logger.debug(f'Classes: {classes}, sizes: {sizes}')
    #image_show(train)
    net = train_net(dataset=train, epochs=11)
    torch.save(net.state_dict(), str(model_path))
    #torch.save(net, str(whole_model_path))


def test():
    logger.debug('start testing')
    _, test, classes, sizes = get_data()
    logger.debug(test.batch_size)
    correct, total = test_net(net_path=model_path, dataset=test)
    message = 'Accuracy of the network on the {0} test images: {1}%'.format(
        len(test),
        100 * correct / total,
    )
    logger.debug(message)


def convert():
    """Приводим модель к типу onnx."""
    change_type(model_path)
