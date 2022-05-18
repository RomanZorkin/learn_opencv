import logging
from pathlib import Path

import cv2
import numpy as np

from recognizer import config, layers, tags

logger = logging.getLogger(__name__)

onnx_model_path = config.MODEL_PATH
images = Path('recognizer/images')
labels = ['cats', 'dogs', 'ground']
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')


def create_net(net_path: Path):
    """Загружаем модель распознования."""
    logger.debug(f'Загружается модель: {net_path}')
    return cv2.dnn.readNetFromONNX(str(net_path))


def create_image(image: Path) -> cv2:
    """Преобразование изображения в объект cv2"""
    logger.debug(f'Текущее изображение: {image.stem}')
    return cv2.imread(str(image))


def recognize():
    for image_path in images.iterdir():
        logger.debug('start _run function')
        net = create_net(onnx_model_path)
        image = create_image(image_path)
        (h, w) = image.shape[:2]
        net_layers = layers.get_layers(net, image)
        logger.debug(f'net_layers: {net_layers}')
        biggest_pred_index = np.array(net_layers).argmax()
        #image_tags = tags.create_tags(image, net_layers)
        #logger.debug(image_tags)
        logger.debug(biggest_pred_index)
        #logger.debug(max(image_tags['confidences']))
