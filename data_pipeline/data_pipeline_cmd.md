cd /d E:\MTech\Sem_3\MLOnEdge

# 1. Activate your Conda Environment
conda activate logibridge_py311

# 2. Check Mosquitto Service Status
sc query mosquitto

# 3. Run Simulator for Normal/No Anomaly Data (30 mins)
python data_pipeline\simulator.py --anomaly none --duration 1800
copy outputs\sensor_logs.csv outputs\sensor_logs_none.csv

# 4. Run Simulator for Temperature Drift Data (30 mins)
python data_pipeline\simulator.py --anomaly temp_drift --duration 1800
copy outputs\sensor_logs.csv outputs\sensor_logs_temp_drift.csv

# 5. Run Simulator for Combined Anomaly Data (30 mins)
python data_pipeline\simulator.py --anomaly combined --duration 1800
copy outputs\sensor_logs.csv outputs\sensor_logs_combined.csv

# 6. Preprocessing & Feature Extraction (Generates normal_, warning_, and critical_features.csv)
python data_pipeline\preprocessing.py

# 7. Compute Normalization Stats (Generates training_stats.npy)
python data_pipeline\normalization.py

# 8. Combine Features into Unified Dataset
python training\generate_dataset.py

# 9. Train MLP Base Model
python training\train_model.py