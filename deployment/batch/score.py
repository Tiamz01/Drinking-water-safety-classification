#!/usr/bin/env python
# coding: utf-8



get_ipython().system('pip freeze | grep scikit-learn')


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report, confusion_matrix, precision_score, recall_score, f1_score

import mlflow
import mlflow.sklearn
from google.cloud import storage as gcs_storage
from google.auth.exceptions import DefaultCredentialsError


# input_file = f'https://github.com/Tiamz01/water__quality_classification_mlops/blob/master/01-data_collection_and_model/data/waterQuality1.csv'
input_file = "01-data_collection_and_model/data/waterQuality.csv"
output_file = f"output/outcome.csv"

RUN_ID = os.getenv('RUN_ID', 'b148d239d6b84b08961892fe7b8dfc95')


def get_and_clean_data(input_file):
    data =  pd.read_csv(input_file)
    data.replace('#NUM!', np.nan, inplace=True)
    data.dropna()
    # Convert all columns to numeric, forcing non-numeric values to NaN
    data = data.apply(pd.to_numeric, errors='coerce')

    cleaned_data = data.dropna()
    df = cleaned_data
    return df



data = get_and_clean_data(input_file)
data



def feature_engineering(df):
    # Features and target variable
    X = df.drop(columns=['is_safe'])
    y = df['is_safe']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test



def load_model(run_id):
    logged_model = f"gs://water_quality_model/1/{run_id}/artifacts/model/gb_model"
    model = mlflow.pyfunc.load_model(logged_model)
    return model


def apply_model(input_file, run_id, output_file):
    df = get_and_clean_data(input_file)
    
    # Check if 'is_safe' column exists
    if 'is_safe' not in df.columns:
        raise KeyError("'is_safe' column is missing from the data.")
    
    X_train, X_test, y_train, y_test = feature_engineering(df)
    
    model = load_model(run_id)
    
    y_pred = model.predict(X_test)
    
    # Save the results
    df_result = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df_result['model_version'] = run_id
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df_result.to_csv(output_file, index=False)
    
    return df_result
    

apply_model(input_file=input_file, run_id=RUN_ID, output_file=output_file)




