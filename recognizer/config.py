from pathlib import Path

model_name = 'wolf'

CONFIDENCE = 0.5
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

MODEL_PATH = Path(f'recognizer/models/{model_name}.onnx')
