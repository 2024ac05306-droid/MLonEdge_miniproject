# 1. Retrain the model using Python 3.11 and tf_keras
& "$env:USERPROFILE\miniconda3\envs\logibridge_py311\python.exe" training/train_model.py

# 2. Run the pruning script
& "$env:USERPROFILE\miniconda3\envs\logibridge_py311\python.exe" training/prune.py