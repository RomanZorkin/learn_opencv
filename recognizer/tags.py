import logging
from typing import Any

import cv2
import numpy as np

from recognizer import config

logger = logging.getLogger(__name__)


def create_tags(image: cv2, layers: cv2) -> dict[str, list[Any]]:
    logger.debug('start create tags')
    tags: dict[str, list[Any]] = {
        'marks': [],
        'confidences': [],
    }
    for detection in layers:
        confidence = abs(abs(detection[0]) - abs(detection[1]))
        logger.debug(f'detection: {detection}, confidence: {confidence}')
        if confidence > config.CONFIDENCE:
            mark = np.argmax(detection)
            logger.debug(f'mark: {mark}')
            tags['marks'].append(mark)
            tags['confidences'].append(confidence)
    return tags
