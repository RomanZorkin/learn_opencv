from pathlib import Path

model = {
    'wolf_tiger': {
        'resolution': (32, 32),
        'scalefactor': 68,
        'confidence': 0.15,
    },
    'cat_dogs': {
        'resolution': (128, 128),
        'scalefactor': 311,
        'confidence': 0.12,
    },
}
model_name = 'cat_dogs'


CONFIDENCE = model[model_name]['confidence']
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

MODEL_PATH = Path(f'recognizer/models/{model_name}.onnx')
RESOLUTION = model[model_name]['resolution']
SCALEFACTOR = 1.0 / model[model_name]['scalefactor']

