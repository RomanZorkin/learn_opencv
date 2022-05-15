import logging

import cv2

logger = logging.getLogger(__name__)


def get_layers(net: cv2, image: cv2) -> cv2:    
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    logger.debug(f'image.shape: {image.shape}, blob.shape {blob.shape}, blob: {blob}')
    
    # усанавливаем blob как вход сети
    net.setInput(blob)
    
    # получаем имена всех слоев    
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    
    return net.forward()
