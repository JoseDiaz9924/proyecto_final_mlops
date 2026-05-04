#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path):
    return pd.read_csv(path)


def preprocess(df, target):
    # eliminar nulos
    df = df.dropna(subset=[target])

    features = [
        "OverallQual",
        "GrLivArea",
        "GarageCars",
        "GarageArea",
        "TotalBsmtSF",
        "1stFlrSF",
        "FullBath",
        "TotRmsAbvGrd",
        "YearBuilt",
        "YearRemodAdd"
    ]

    X = df[features]
    y = df[target]

    # Escalamiento
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


def split_data(X, y, test_size, random_state):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

