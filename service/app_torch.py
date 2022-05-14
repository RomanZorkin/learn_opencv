import cv2
import matplotlib.pyplot as plt

from service.utils import detect_objects, plot_boxes, load_class_names
from service.darknet import Darknet

def run():
    # Установить порог NMS
    nms_threshold = 0.6
    # Установить порог IoU
    iou_threshold = 0.4
    cfg_file = "service/cfg/yolov3.cfg"
    weight_file = "service/weights/yolov3.weights"
    namesfile = "service/data/coco.names"
    m = Darknet(cfg_file)
    m.load_weights(weight_file)
    class_names = load_class_names(namesfile)
    # m.print_network()
    original_image = cv2.imread("service/images/morj_2.jpg")
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(original_image, (m.width, m.height))
    # обнаруживаем объекты
    boxes = detect_objects(m, img, iou_threshold, nms_threshold)
    # вычерчиваем изображение с ограничивающими рамками и соответствующими метками классов объектов
    plot_boxes(original_image, boxes, class_names, plot_labels=True)