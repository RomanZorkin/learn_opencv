import logging
from pathlib import Path

import cv2
import numpy as np

from recognizer import config, draw, layers, tags

logger = logging.getLogger(__name__)

onnx_model_path = config.MODEL_PATH
images = Path('recognizer/images')
labels = ['cats', 'dogs', 'ground']
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')


def create_net(net_path: Path) -> cv2:
    """Загружаем модель распознования."""
    logger.debug(f'Загружается модель: {net_path}')
    return cv2.dnn.readNetFromONNX(str(net_path))


def download_image(image: Path) -> cv2:
    """Преобразование изображения в объект cv2"""
    logger.info(f'Текущее изображение: {image.stem}')
    return cv2.imread(str(image))


def recognize():
    logger.info('\n\nstart _run function')
    net = create_net(onnx_model_path)

    for image_path in images.iterdir():
        if not image_path.suffix:
            continue
        image = download_image(image_path)
        (h, w) = image.shape[:2]

        net_layers = layers.get_layers(net, image)
        image_tags = tags.create_tags(image, net_layers)

        logger.info(image_tags)
        if not image_tags['label']:
            continue
        draw.draw(image, image_tags)
