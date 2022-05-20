from pathlib import Path

<<<<<<< HEAD
animal = 'squirrel'
=======
animal = 'wolf_tiger'
>>>>>>> a4f0d7d7af11f0b077991a6dd4b49555e5a736ee

MODEL_PATH = Path(f'simple_net_cnn/models/{animal}.pth')
WHOLE_MODEL_PATH = Path(f'simple_net_cnn/models/{animal}.pt')

MODEL_NAME = f'{animal}.onnx'

<<<<<<< HEAD
DATASET_PATH = Path('simple_net_cnn/data/animals/raw-img', animal)
TRAIN_PATH = Path(DATASET_PATH, 'training_set')
TEST_PATH = Path(DATASET_PATH, 'test_set')
=======
DATASET_PATH = Path(f'simple_net_cnn/data/{animal}')
TRAIN_PATH = Path(DATASET_PATH, 'train')
TEST_PATH = Path(DATASET_PATH, 'test')
>>>>>>> a4f0d7d7af11f0b077991a6dd4b49555e5a736ee
