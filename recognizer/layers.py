import logging

import cv2
import numpy as np

from recognizer.config import RESOLUTION, SCALEFACTOR

logger = logging.getLogger(__name__)


def get_layers(net: cv2, image: cv2) -> np.ndarray:
    # Функция возвращает массив слоев по изображению
    blob = cv2.dnn.blobFromImage(image, SCALEFACTOR, RESOLUTION, swapRB=True)
    logger.debug(f'image.shape: {image.shape}, blob.shape {blob.shape}')

    # усанавливаем blob как вход сети
    net.setInput(blob)

    return net.forward()
