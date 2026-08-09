# MLonEdge_miniproject
This Repository is created for assignment work of ML on Edge. 


# Group 44
| Name | BITS ID |
|:-----|:------:|
|Hirdalappa H|2024AC05306|
|Pranav S|2024AC05355|
|Shashank Shekhar|2024AC05773|
|Suresh M|2024AC05271|

**Demo Video link** : https://drive.google.com/drive/folders/19dGMozpOk2iO4Po7tHCpLYYPYgw-IOjo?usp=sharing


# Goal Of the assignment
The goal of this assignment is to design and implement an Edge AI-based cold-chain monitoring system for refrigerated trucks that can operate reliably without continuous internet connectivity. The system aims to monitor temperature, vibration, and door events in real time, classify the operational state of the cargo, and generate immediate alerts for anomalies. It also demonstrates end-to-end Edge AI deployment through model optimization, Docker containerization, MQTT communication, and MLOps-based monitoring for scalable fleet management.


# Tool stack and Librabies
This tool stack aligns with the assignment components, including Edge AI model development, MQTT-based communication, Docker deployment, MLOps monitoring, model optimization, and automated deployment.
| Tool Name | Purpose |
|:-----|:------:|
|Git | version control|
|Git Hub Actions | CI/CD |
|VS code|Model Development|
|Source tree|Visual Git flow|
|Python |Programming Language Python3.11|
|Pandas and numpy |Data Processing and Feature Enginerring|
|Matplotlib |Exploratory Data Analysis (EDA) and result visualization|
|MLflow| Model registry and log and Experiment tracking|
|Docker Hub |Containarization and Deployment|
|MQTT, Mosquitto Broker |Sensor data transmission and inference messaging|
|Ansible |Automated model deployment and OTA updates|
|Pytest |Unit and integration testing|


## Project flow 
                MQTT Sensor Simulation
                         │
                         ▼
               simulator.py
                         │
        Generates sensor logs (.csv)
                         │
                         ▼
                 outputs/
     ├── sensor_logs_none.csv
     ├── sensor_logs_temp_drift.csv
     └── sensor_logs_combined.csv
                         │
                         ▼
              preprocessing.py
                         │
        Sliding Window (30 s)
        Moving Average Filter
        Feature Extraction
                         │
                         ▼
                   data/
     ├── normal_features.csv
     ├── warning_features.csv
     └── critical_features.csv
                         │
                         ▼
            generate_dataset.py
                         │
     Merge + Label + Shuffle Dataset
                         │
                         ▼
           data/training_dataset.csv
                         │
                         ▼
              normalization.py
                         │
      Calculate Mean & Std (Normal Class)
                         │
                         ▼
      data/training_stats.npy
                         │
                         ▼
               train_model.py
               data/training_stats.npy
                         │
     Train → Validate → Save Model
                         │
                         ▼
        models/best_model.keras
        models/best_model_pruned.keras
                         │
                         ▼
           inference_service.py (directly loads model_ptq.tflite via tflite_runtime.interpreter.Interpreter)
                         │
                         ▼
            models/model_ptq.tflite
                         │
                         ▼
            inference/inference_service.py
                         │
                         ▼
            deployment/logibridge_deploy.yml
                         │
                         ▼
     Normalize → Predict → Alarm
