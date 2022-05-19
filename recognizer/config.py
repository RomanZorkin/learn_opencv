from pathlib import Path

model = {
    'wolf_tiger': {
        'resolution': (32, 32),
        'scalefactor': 68,
        'confidence': 0.15,
        'labels': ('tiger', 'wolf')
    },
    'cat_dogs': {
        'resolution': (229, 229),
        'scalefactor': 100,
        'confidence': 0.10,
        'labels': ('cat', 'dog')
    },
}
model_name = 'wolf_tiger'


CONFIDENCE = model[model_name]['confidence']
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

MODEL_PATH = Path(f'recognizer/models/{model_name}.onnx')
RESOLUTION = model[model_name]['resolution']
SCALEFACTOR = 1.0 / model[model_name]['scalefactor']
LABELS = model[model_name]['labels']
