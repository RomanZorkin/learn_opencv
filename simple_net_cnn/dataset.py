from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from simple_net_cnn import config


def get_data() -> tuple[DataLoader, DataLoader, list[str], dict[str, int]]:
    """Подготавливаем данные для обработки.

    Из дирректории с датасетом указываются два пути на набор данных для обучения и для тестирования
    Задаются правила трансформации картинок. Формируются соответствующие объекты DataLoader
    """
    train_path = config.TRAIN_PATH
    test_path = config.TEST_PATH

    transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    train_set = datasets.ImageFolder(str(train_path), transform=transform)
    test_set = datasets.ImageFolder(str(test_path), transform=transform)

    train = DataLoader(train_set, batch_size=32, shuffle=True)
    test = DataLoader(test_set, batch_size=32, shuffle=True)

    classes = train_set.classes
    sizes = {
        'train': len(train_set),
        'test': len(test_set),
    }

    return train, test, classes, sizes
