import logging
from pathlib import Path

import cv2
import numpy as np

from recognizer import layers, tags

logger = logging.getLogger(__name__)

onnx_model_path = Path('recognizer/models/cat_dog.onnx')
image_path = Path('recognizer/images/cat.jpg')
labels = ['cats', 'dogs', 'ground']
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

def create_net(net_path: Path):
    """Загружаем модель распознования."""
    return cv2.dnn.readNetFromONNX(str(onnx_model_path))


def create_image(image: Path) -> cv2:
    """Преобразование изображения в объект cv2"""
    return cv2.imread(str(image))


def run():
    logger.debug('start _run function')
    net = create_net(onnx_model_path)
    image = create_image(image_path)    
    (h, w) = image.shape[:2]
    net_layers = layers.get_layers(net, image)
    logger.debug(f'net_layers: {net_layers}')
    image_tags = tags.create_tags(image, net_layers)
    logger.debug(image_tags)
    logger.debug(max(image_tags['confidences']))
