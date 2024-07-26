#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.pyfunc
from google.auth.exceptions import DefaultCredentialsError

# Set the environment variable for Google Cloud credentials
if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
    raise EnvironmentError("GOOGLE_APPLICATION_CREDENTIALS environment variable not set.")


def get_and_clean_data(input_file):
    data = pd.read_csv(input_file)
    data.replace('#NUM!', np.nan, inplace=True)
    data.dropna()
    data = data.apply(pd.to_numeric, errors='coerce')
    cleaned_data = data.dropna()
    return cleaned_data

def feature_engineering(df):
    X = df.drop(columns=['is_safe'])
    y = df['is_safe']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def load_model(run_id):
    logged_model = "gs://water_quality_model/1/{run_id}/artifacts/model/gb_model"
    try:
        model = mlflow.pyfunc.load_model(logged_model)
    except DefaultCredentialsError as e:
        print("Credentials error:", e)
        raise
    return model

def apply_model(input_file, run_id, output_file):
    df = get_and_clean_data(input_file)
    
    if 'is_safe' not in df.columns:
        raise KeyError("'is_safe' column is missing from the data.")
    
    X_train, X_test, y_train, y_test = feature_engineering(df)
    
    model = load_model(run_id)
    
    y_pred = model.predict(X_test)
    
    df_result = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df_result['model_version'] = run_id
    
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df_result.to_csv(output_file, index=False)
    
    return df_result

if __name__ == '__main__':
    RUN_ID = 'b148d239d6b84b08961892fe7b8dfc95'
    input_file = 'https://raw.githubusercontent.com/Tiamz01/data_repo/main/waterQuality1.csv'
    output_file = 'output/predictions.csv'

    apply_model(input_file=input_file, run_id=RUN_ID, output_file=output_file)
