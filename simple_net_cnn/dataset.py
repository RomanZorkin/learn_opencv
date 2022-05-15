from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data() -> tuple[DataLoader, DataLoader]:
    """Подготавливаем данные для обработки.

    Из дирректории с датасетом указываются два пути на набор данных для обучения и для тестирования
    Задаются правила трансформации картинок. Формируются соответствующие объекты DataLoader
    """
    train_path = Path('simple_net_cnn/data/dataset/training_set')
    test_path = Path('simple_net_cnn/data/dataset/test_set')

    transform = transforms.Compose([
        transforms.RandomResizedCrop(128),
        transforms.ToTensor(),
    ])

    train_set = datasets.ImageFolder(str(train_path), transform=transform)
    test_set = datasets.ImageFolder(str(test_path), transform=transform)

    train = DataLoader(train_set, batch_size=32, shuffle=True)
    test = DataLoader(test_set, batch_size=32, shuffle=True)

    return train, test
