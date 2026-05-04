#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importar paquetes
import os
from pathlib import Path
import import_ipynb
import yaml
import sys
import traceback
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from preprocess import load_data, preprocess, split_data


# In[2]:


# Definicion de rutas
workspace_dir  = os.getcwd()
mlruns_dir     = os.path.join(workspace_dir, "mlruns")
tracking_uri   = "file:///" + os.path.abspath(mlruns_dir).replace("\\", "/")
artifact_loc   = tracking_uri          # experimentos y modelos en el mismo directorio
model_pkl_path = os.path.join(workspace_dir, "model.pkl")
data_path_dir = os.path.join(workspace_dir,"data")
config_path = os.path.join(workspace_dir, "config.yaml")

print(f"[train] CWD            : {workspace_dir}")
print(f"[train] MLRuns dir     : {mlruns_dir}")
print(f"[train] Tracking URI   : {tracking_uri}")
print(f"[train] Data           : {data_path_dir}")
print(f"[train] Config         : {config_path}")
os.makedirs(mlruns_dir, exist_ok=True)


# In[5]:


# Definhicion del tracking de mlflow
mlflow.set_tracking_uri(tracking_uri)

experiment_name = "CI-CD-Lab-MLflow"
try:
    experiment_id = mlflow.create_experiment(
        name=experiment_name,
        artifact_location=artifact_loc,
    )
    print(f"[train] Experimento creado  → ID: {experiment_id}")
except mlflow.exceptions.MlflowException as exc:
    if "RESOURCE_ALREADY_EXISTS" not in str(exc):
        raise
    exp = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = exp.experiment_id
    print(f"[train] Experimento existente → ID: {experiment_id}")


# In[7]:


# Parametros globales
with open(config_path, "r") as f:
    config = yaml.safe_load(f)


# In[9]:


data_path = os.path.join(data_path_dir,"dataset.csv")


# In[11]:


with mlflow.start_run(experiment_id=experiment_id) as run:

    # cargar datos
    df = load_data(data_path) 
    X, y, scaler = preprocess(df, config["data"]["target"])

    X_train, X_test, y_train, y_test = split_data(
        X, y,
        config["model"]["test_size"],
        config["model"]["random_state"]
    )

    # modelo
    model = LinearRegression()
    model.fit(X_train, y_train)

    # predicción
    y_pred = model.predict(X_test)

    # métricas
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # logging MLflow
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_metric("mse", rmse)
    mlflow.log_metric("r2", r2)

    mlflow.sklearn.log_model(
        model,
        "model",
        input_example=X_test[:5]
    )

    print(f"MSE: {mse}")
    print(f"R2: {r2}")


# In[13]:


joblib.dump(model, model_pkl_path)
print(f"[train] model.pkl guardado en: {model_pkl_path}")
print("[train] ✅ Entrenamiento completado exitosamente.")

