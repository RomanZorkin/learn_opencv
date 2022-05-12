import os
import time
import sys

import cv2
import numpy as np

CONFIDENCE = 0.5
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

# конфигурация нейронной сети
config_path = 'cfg/yolov3.cfg'
# файл весов сети YOLO
weights_path = 'weights/yolov3.weights'
# weights_path = "weights/yolov3-tiny.weights"

# загрузка всех меток классов (объектов)
with open('data/coco.names', 'r') as f:
    labels = f.read().strip().split('\n')
#labels = open("data/coco.names").read().strip().split("\n")
# генерируем цвета для каждого объекта и последующего построения
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')


