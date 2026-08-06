import json
import os
from pathlib import Path
import sys
import numpy as np
import paho.mqtt.client as mqtt

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variable with default fallback
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/model_ptq.tflite")
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))

# Load TFLite Interpreter
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

print(f"[INFO] Loading TFLite model from: {MODEL_PATH}")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASS_NAMES = ["Normal", "Warning", "Critical"]


def preprocess_data(raw_features):
    """Normalize input features using saved stats or standard arrays."""
    # Input shape expected: (1, 6)
    data = np.array(raw_features, dtype=np.float32)
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    
    # Handle INT8 input quantization if model requires int8
    if input_details[0]['dtype'] == np.int8:
        scale, zero_point = input_details[0]['quantization']
        data = (data / scale + zero_point).astype(np.int8)
        
    return data


def run_inference(data):
    """Run TFLite model inference."""
    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    # Dequantize if output is INT8
    if output_details[0]['dtype'] == np.int8:
        scale, zero_point = output_details[0]['quantization']
        output = (output.astype(np.float32) - zero_point) * scale

    pred_class_idx = int(np.argmax(output, axis=1)[0])
    confidence = float(np.max(output))
    return CLASS_NAMES[pred_class_idx], confidence


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        truck_id = payload.get("truck_id", "TRUCK_UNKNOWN")
        features = payload.get("features") # Expected list of 6 numbers

        if features and len(features) == 6:
            input_tensor = preprocess_data(features)
            label, confidence = run_inference(input_tensor)

            # Publish result to specified topic structure
            pub_topic = f"logibridge/trucks/{truck_id}/inference"
            result_payload = {
                "truck_id": truck_id,
                "status": label,
                "confidence": round(confidence, 4),
                "model_used": Path(MODEL_PATH).name,
            }

            client.publish(pub_topic, json.dumps(result_payload))
            print(f"[INFERENCE] Truck: {truck_id} | Result: {label} ({confidence:.2f}) -> Published to {pub_topic}")

    except Exception as e:
        print(f"[ERROR] Failed to process message: {e}")


def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = on_message
    
    print(f"[MQTT] Connecting to {MQTT_BROKER}:{MQTT_PORT}...")
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    
    # Subscribe to telemetry from all trucks
    client.subscribe("logibridge/trucks/+/telemetry")
    print("[MQTT] Subscribed to logibridge/trucks/+/telemetry. Listening...")
    
    client.loop_forever()


if __name__ == "__main__":
    main()