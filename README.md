# learn_opencv

В каталоге __service__ находится простой рабочий модуль (модель)распознования YOLO. Существует два варианта запуска:
1. без pytorch
2. с pytorch

для выбора варианта в файле main.py необходимо переменной variant присвоить соответствующее значение (1 или 2)

В каталоге __simple_net__ находится простоя нейронная сеть для создания моделей

В каталоге __simple_net_cnn__ находится простоя нейронная сеть для создания моделей для работы с изображениями.

для запуска обучения:
с помощью команды в терминале
```bash
python -m simple_net_cnn run
```
с помощью Makefile
```bash
make train_cnn_net
```
для запуска теста:
с помощью команды в терминале
```bash
python -m simple_net_cnn test
```
с помощью Makefile
```bash
make test_cnn_net
```
для конвертации модели в формат onnx:
с помощью команды в терминале
```bash
python -m simple_net_cnn convert
```
с помощью Makefile
```bash
make convert_cnn_net
```

В каталоге __recognizer__ находится простой модуль для распознования изоражений
целевое изображение находится в каталоге images. До начала работы в каталог models необходимо поместить
модель с расширением .onnx. В файле config.py записаны настройки каждой модели, для запуска соответствующей переменной
model_name необходимо присвоить нужное значение. У каждрй моедли есть обязательные характеристики:
resolution - разрешение картинок по которым происходило обучение модели.
scalefactor - значение для расчета каналов распознования изображения (пока задается методом тыка, обычно разрешение * 2 и смотрим качество отбора, далее меняем, в процессе разработки модуль для автоподстчета)
confidence - значение вероятности, также как и предыдущий подбирается, для каждой модели
Команды для запуска:
Запуск виртуального окружения
```bash
source venv/bin/activate
```
запуск приложения с помощью Makefile
```bash
make recognize
```
запуск приложения в командной строке
```bash
python -m recognizer
```
## Репозитарий с датасетом
https://disk.yandex.ru/d/f9RHNOecdMooTQ

## Ссылки на обучающий материал

1. В этом уроке вы узнаете, как найти и обнаружить объекты с помощью современной техники YOLO v3 с OpenCV или PyTorch в Python
https://waksoft.susu.ru/2021/05/19/kak-vypolnit-obnaruzhenie-obektov-yolo-s-pomoshhyu-opencv-i-pytorch-v-python/https://waksoft.susu.ru/2021/05/19/kak-vypolnit-obnaruzhenie-obektov-yolo-s-pomoshhyu-opencv-i-pytorch-v-python/

для получения файла весов wget https://pjreddie.com/media/files/yolov3.weights

с сайта YOLO project https://pjreddie.com/darknet/yolo/


2. статья про opencv
https://habr.com/ru/post/519454/

3. MASK-RCNN для поиска крыш по снимкам с беспилотников
https://habr.com/ru/company/lanit/blog/500752/

4. статья про pytorch https://neurohive.io/ru/tutorial/glubokoe-obuchenie-s-pytorch/
и репозитарий к ней https://github.com/adventuresinML/adventures-in-ml-code/blob/master/pytorch_nn.py

5. статья pytorch cnn модель для обработки изображений
https://towardsdatascience.com/how-to-apply-a-cnn-from-pytorch-to-your-images-18515416bba1
и репозитарий с кодом https://github.com/alexkhrustalev/DS_projects/blob/master/cats_and_dogs.py

6. статья как склеить свою обученную модель и opencv https://habr.com/ru/post/478208/ и еще https://habr.com/ru/post/494804/
