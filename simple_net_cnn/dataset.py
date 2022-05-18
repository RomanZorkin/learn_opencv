from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from simple_net_cnn import config


def get_data() -> tuple[DataLoader, DataLoader]:
    """Подготавливаем данные для обработки.

    Из дирректории с датасетом указываются два пути на набор данных для обучения и для тестирования
    Задаются правила трансформации картинок. Формируются соответствующие объекты DataLoader
    """
    train_path = config.TRAIN_PATH
    test_path = config.TEST_PATH

    transform = transforms.Compose([
        transforms.RandomResizedCrop(128),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_set = datasets.ImageFolder(str(train_path), transform=transform)
    test_set = datasets.ImageFolder(str(test_path), transform=transform)

    train = DataLoader(train_set, batch_size=32, shuffle=True)
    test = DataLoader(test_set, batch_size=32, shuffle=True)

    return train, test
