import pickle
import os
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from flask import Flask
import request, jsonify


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


def load_model():
    try:
        model_uri = "gs://water_quality_model/1/b148d239d6b84b08961892fe7b8dfc95/artifacts/model/gb_model"
        model = mlflow.sklearn.load_model(model_uri)
        print("Model loaded from GCS.")
    except Exception as e:
        print(f"Error loading model from GCS: {e}")
        with open('best_log_reg.bin', 'rb') as f_in:
            model = pickle.load(f_in)
            print("Model loaded from local binary file.")
    return model

model = load_model()

def apply_model(input_file, output_file):
    df = get_and_clean_data(input_file)
    
    if 'is_safe' not in df.columns:
        raise KeyError("'is_safe' column is missing from the data.")
    
    X_train, X_test, y_train, y_test = feature_engineering(df)
    model = load_model()
    y_pred = model.predict(X_test)
    df_result = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    # df_result['model_version'] = run_id

    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df_result.to_csv(output_file, index=False)

    return df_result


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    df = get_and_clean_data(data)
    
    if 'is_safe' not in df.columns:
        return jsonify({'error': "'is_safe' column is missing from the data."}), 400
    
    X = feature_engineering(df)
    y_pred = model.predict(X).tolist()
    
    response = {
        "predictions": y_pred,
        "message": ["water is safe for drinking" if pred == 1 else "water is not safe for drinking" for pred in y_pred]
    }

    return jsonify(response)

if __name__ == '__main__':
    # RUN_ID = 'b148d239d6b84b08961892fe7b8dfc95'
    input_file = 'https://raw.githubusercontent.com/Tiamz01/data_repo/main/waterQuality1.csv'
    output_file = 'output/predictions.csv'

    apply_model(input_file=input_file, run_id=RUN_ID, output_file=output_file)
