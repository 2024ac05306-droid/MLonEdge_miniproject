# execution sequence

```
cd /d E:\MTech\Sem_3\MLOnEdge

.venv\Scripts\activate

sc query mosquitto

python data_pipeline\simulator.py --anomaly none --duration 1800
copy outputs\sensor_logs.csv outputs\sensor_logs_none.csv

python data_pipeline\simulator.py --anomaly temp_drift --duration 1800
copy outputs\sensor_logs.csv outputs\sensor_logs_temp_drift.csv

python data_pipeline\simulator.py --anomaly combined --duration 1800
copy outputs\sensor_logs.csv outputs\sensor_logs_combined.csv

python data_pipeline\preprocessing.py

python data_pipeline\normalization.py

python training\generate_dataset.py

python training\train_model.py
```



# Task C2 pipeline:
```
Simulator
   ↓
Raw sensor CSV
   ↓
Preprocessing + Feature Extraction
   ↓
normal_features.csv
warning_features.csv
critical_features.csv
   ↓
Normalization
   ↓
training_stats.npy
   ↓
Dataset generation
   ↓
MLP Training
```
