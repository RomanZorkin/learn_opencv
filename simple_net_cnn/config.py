from pathlib import Path

animal = 'wolf'

MODEL_PATH = Path(f'simple_net_cnn/models/{animal}.pth')
WHOLE_MODEL_PATH = Path(f'simple_net_cnn/models/{animal}.pt')

MODEL_NAME = f'{animal}.onnx'

DATASET_PATH = Path(f'simple_net_cnn/data/{animal}')
TRAIN_PATH = Path(DATASET_PATH, 'train')
TEST_PATH = Path(DATASET_PATH, 'test')
