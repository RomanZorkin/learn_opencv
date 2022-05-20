import logging

import cv2
import numpy as np

from recognizer.config import RESOLUTION, SCALEFACTOR

logger = logging.getLogger(__name__)


<<<<<<< HEAD
def get_layers(net: cv2, image: cv2) -> cv2:    
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (128, 128), swapRB=True, crop=False)
    logger.debug(f'image.shape: {image.shape}, blob.shape {blob.shape}')
    
    # усанавливаем blob как вход сети
    net.setInput(blob)
    
    # получаем имена всех слоев    
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    logger.debug(f'Имена слоев: {ln}')
=======
def get_layers(net: cv2, image: cv2) -> np.ndarray:
    # Функция возвращает массив слоев по изображению
    blob = cv2.dnn.blobFromImage(image, SCALEFACTOR, RESOLUTION, swapRB=True)
    logger.debug(f'image.shape: {image.shape}, blob.shape {blob.shape}')

    # усанавливаем blob как вход сети
    net.setInput(blob)
>>>>>>> a4f0d7d7af11f0b077991a6dd4b49555e5a736ee

    return net.forward()
