# 1. Retrain the model using Python 3.11 and tf_keras
& "$env:USERPROFILE\miniconda3\envs\logibridge_py311\python.exe" training/train_model.py

# 2. Run the pruning script
& "$env:USERPROFILE\miniconda3\envs\logibridge_py311\python.exe" training/prune_quantise.py

# 3. Convert script
& "$env:USERPROFILE\miniconda3\envs\logibridge_py311\python.exe" training/convert_ptq.py

# 4. Normalization experiment 
& "$env:USERPROFILE\miniconda3\envs\logibridge_py311\python.exe" training/run_norm_experiment.py
