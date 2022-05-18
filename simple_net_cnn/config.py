from pathlib import Path

animal = 'squirrel'

MODEL_PATH = Path(f'simple_net_cnn/models/{animal}.pth')
WHOLE_MODEL_PATH = Path(f'simple_net_cnn/models/{animal}.pt')

MODEL_NAME = f'{animal}.onnx'

DATASET_PATH = Path('simple_net_cnn/data/animals/raw-img', animal)
TRAIN_PATH = Path(DATASET_PATH, 'training_set')
TEST_PATH = Path(DATASET_PATH, 'test_set')
