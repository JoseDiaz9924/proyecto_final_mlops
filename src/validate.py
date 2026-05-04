#!/usr/bin/env python
# coding: utf-8

# In[65]:


import os
from pathlib import Path
import joblib
import yaml
import import_ipynb
from preprocess import load_data, preprocess, split_data
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# In[67]:


# rutas
THRESHOLD = 0.6
workspace_dir  = Path.cwd().resolve().parent
data_path_dir = os.path.join(workspace_dir,"data")
config_path = os.path.join(workspace_dir, "config.yaml")
model_pkl_path = os.path.join(workspace_dir, "model.pkl")


# In[69]:


# Cargar el modelo 
print(f"[validate] Buscando modelo en: {model_pkl_path}")

if not os.path.exists(model_pkl_path):
    print(f"[validate] ❌ ERROR: No se encontró model.pkl.")
    print(f"[validate]    Archivos en CWD: {os.listdir(os.getcwd())}")
    sys.exit(1)

model = joblib.load(model_pkl_path)
print(f"[validate] Modelo cargado. Tipo: {type(model).__name__}")
print(f"[validate] Features esperadas: {model.n_features_in_}")


# In[71]:


with open(config_path, "r") as f:
    config = yaml.safe_load(f)


# In[73]:


data_path = os.path.join(data_path_dir,"dataset.csv")


# In[81]:


df = load_data(data_path) 
X, y, scaler = preprocess(df, config["data"]["target"])

X_train, X_test, y_train, y_test = split_data(
    X, y,
    config["model"]["test_size"],
    config["model"]["random_state"]
)

r2 = r2_score(y_test, y_pred)
print(f"[validate] X_test shape: {X_test.shape}")


# In[79]:


if r2 >= THRESHOLD:
    print("[validate] ✅ PASA quality gate")
else:
    raise ValueError("[validate] ❌ NO pasa quality gate")

