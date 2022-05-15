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
        confidence = abs(detection[0] - detection[1])
        if confidence > config.CONFIDENCE:
            mark = np.argmax(detection)
            tags['marks'].append(mark)
            tags['confidences'].append(confidence)
    return tags
