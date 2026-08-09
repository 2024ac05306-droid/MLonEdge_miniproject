import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


import tensorflow.lite as tflite

import sys
import time
from pathlib import Path
import numpy as np

# Set project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# TFLite Import Fallback
try:
    import tflite_runtime.interpreter as tflite  # type: ignore
except ImportError:
    import tensorflow.lite as tflite

from config import DATASET_FILE, TRAINING_STATS, MODEL_DIR
from utils import load_training_dataset, get_features_and_labels, load_training_stats, normalize_features

# 1. Load Data
df = load_training_dataset(DATASET_FILE)
X_raw, y = get_features_and_labels(df)

# 2. Normalize
mean, std = load_training_stats(TRAINING_STATS, DATASET_FILE)
X_norm = normalize_features(X_raw, mean, std)

# 3. Load Recommended Variant (M2: PTQ INT8)
model_path = MODEL_DIR / "model_ptq.tflite"
interpreter = tflite.Interpreter(model_path=str(model_path))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

y_preds = []

# 4. Run TFLite Inference
for sample in X_norm:
    data = np.expand_dims(sample, axis=0).astype(np.float32)

    # Input quantization check
    if input_details[0]["dtype"] == np.int8:
        scale, zero_point = input_details[0]["quantization"]
        data = np.round(data / scale + zero_point)
        data = np.clip(data, -128, 127).astype(np.int8)

    interpreter.set_tensor(input_details[0]["index"], data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    y_preds.append(np.argmax(output))

# 5. Compute Class 2 Recall
cm = confusion_matrix(y, y_preds)
class2_recall = cm[2, 2] / np.sum(cm[2, :])

print("\n" + "=" * 50)
print(f"Recommended Variant (M2 PTQ INT8) Class 2 Recall: {class2_recall:.4f} ({class2_recall * 100:.2f}%)")
print("=" * 50)