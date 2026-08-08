import sys

import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
import tensorflow.lite as tflite

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import get_features_and_labels, load_training_dataset, load_training_stats, normalize_features
from config import DATASET_FILE, MODEL_DIR

def evaluate_model_on_data(interpreter, X_data, y_true):
    """Runs inference and calculates Accuracy and Class 2 Recall."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    y_pred = []
    for sample in X_data:
        sample_input = np.expand_dims(sample, axis=0).astype(np.float32)
        
        # Quantize input if INT8 model
        if input_details[0]['dtype'] == np.int8:
            scale, zero_point = input_details[0]['quantization']
            sample_input = np.clip(np.round(sample_input / scale + zero_point), -128, 127).astype(np.int8)

        interpreter.set_tensor(input_details[0]['index'], sample_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        # Dequantize output if INT8 model
        if output_details[0]['dtype'] == np.int8:
            scale, zero_point = output_details[0]['quantization']
            output = (output.astype(np.float32) - zero_point) * scale

        pred = np.argmax(output, axis=1)[0]
        y_pred.append(pred)

    acc = accuracy_score(y_true, y_pred) * 100.0
    c2_rec = recall_score(y_true, y_pred, labels=[2], average=None, zero_division=0)[0] * 100.0
    return acc, c2_rec

if __name__ == "__main__":
    # 1. Load Data and Base Stats
    df = load_training_dataset(DATASET_FILE)
    X_raw, y = get_features_and_labels(df)
    mean, std = load_training_stats()

    # Split into clean validation set
    _, X_val_raw, _, y_val = train_test_split(
        X_raw, y, test_size=0.20, random_state=42, stratify=y
    )

    # 2. Standard Normalization (Correct Stats)
    X_val_correct = normalize_features(X_val_raw, mean=mean, std=std)

    # 3. Apply +3σ Mean Shift (Shifted Stats Experiment)
    shifted_mean = mean + (3.0 * std)
    X_val_shifted = normalize_features(X_val_raw, mean=shifted_mean, std=std)

    # 4. Benchmark Models on Both Sets
    model_path = MODEL_DIR / "model_ptq.tflite"  # Recommended M2 Variant
    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    acc_correct, rec_correct = evaluate_model_on_data(interpreter, X_val_correct, y_val)
    acc_shifted, rec_shifted = evaluate_model_on_data(interpreter, X_val_shifted, y_val)

    print("=" * 60)
    print("MANDATORY EXPERIMENT: OUT-OF-DISTRIBUTION (3σ SHIFT)")
    print("=" * 60)
    print(f"Correct Stats Baseline  -> Accuracy: {acc_correct:.2f}% | Class 2 Recall: {rec_correct:.2f}%")
    print(f"+3σ Shifted Stats       -> Accuracy: {acc_shifted:.2f}% | Class 2 Recall: {rec_shifted:.2f}%")
    print(f"Absolute Accuracy Drop  -> {acc_correct - acc_shifted:.2f}%")
    print("=" * 60)