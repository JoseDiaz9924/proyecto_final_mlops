# Proyecto MLops

## Descripción
Modelo de regresión con MLflow y CI/CD. 

## Problema 
Se requiere desarrollar un modelo para estimar los precios de venta de viviendas de acuerdo a sus caracteristicas

## Data

La base de datos cuenta con 81 variables cada una con un total de 1460 observaciones

## Estructura del proyecto
project/ │ ├── src/ │ ├── train.py │ ├── validate.py │ └── preprocessing.py │ ├── data/ │ └── dataset.csv │ ├── .github/ │ └── workflows/ │ └── mlflow-ci.yaml │ └── mlruns/
