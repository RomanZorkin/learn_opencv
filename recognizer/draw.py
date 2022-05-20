<<<<<<< HEAD
def draw(layers):
    pass
=======
import logging
from typing import Any
import cv2

logger = logging.getLogger(__name__)


def draw(image: cv2, tags: dict[str: Any]) -> None:
    font_scale = 1
    thickness = 2
    text = str(tags['label'][0])
    cv2.putText(
        image, text, (0, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        color=(0, 0, 0),
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    new_image_path = f'recognizer/images/output/{text}{tags["confidences"][0]}.jpg'
    cv2.imwrite(new_image_path, image)
>>>>>>> a4f0d7d7af11f0b077991a6dd4b49555e5a736ee
