import logging
from pathlib import Path

import cv2
import numpy as np

<<<<<<< HEAD
from recognizer import config, layers, tags
=======
from recognizer import config, draw, layers, tags
>>>>>>> a4f0d7d7af11f0b077991a6dd4b49555e5a736ee

logger = logging.getLogger(__name__)

onnx_model_path = config.MODEL_PATH
images = Path('recognizer/images')
labels = ['cats', 'dogs', 'ground']
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')


<<<<<<< HEAD
def create_net(net_path: Path):
=======
def create_net(net_path: Path) -> cv2:
>>>>>>> a4f0d7d7af11f0b077991a6dd4b49555e5a736ee
    """Загружаем модель распознования."""
    logger.debug(f'Загружается модель: {net_path}')
    return cv2.dnn.readNetFromONNX(str(net_path))


def download_image(image: Path) -> cv2:
    """Преобразование изображения в объект cv2"""
<<<<<<< HEAD
    logger.debug(f'Текущее изображение: {image.stem}')
=======
    logger.info(f'Текущее изображение: {image.stem}')
>>>>>>> a4f0d7d7af11f0b077991a6dd4b49555e5a736ee
    return cv2.imread(str(image))


def recognize():
<<<<<<< HEAD
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
=======
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
>>>>>>> a4f0d7d7af11f0b077991a6dd4b49555e5a736ee
