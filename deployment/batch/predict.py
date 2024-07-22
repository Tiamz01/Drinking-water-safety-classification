import pickle
import os
import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from flask import Flask, jsonify, request

def get_and_clean_data(input_file, is_json=True):
    if is_json:
        # Convert JSON to DataFrame
        df = pd.json_normalize(input_file)
    else:
        # Convert CSV to DataFrame
        df = pd.read_csv(input_file)

    df.replace('#NUM!', np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    cleaned_data = df.dropna()
    return cleaned_data

def feature_engineering(df, is_prediction=True):
    X = df.drop(columns=['is_safe'])
    y = df['is_safe']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if is_prediction:
        # Return only the scaled features for prediction
        return X_scaled
    else:
        # Perform train-test split if not just prediction
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test

def load_model():
    try:
        print('Loading model from GCS...')
        model_uri = "gs://water_quality_model/1/b148d239d6b84b08961892fe7b8dfc95/artifacts/model/gb_model"
        model = mlflow.sklearn.load_model(model_uri)
        print("Model loaded from GCS.")
    except Exception as e:
        print(f"Error loading model from GCS: {e}. Now loading from local binary file...")
       
        with open('best_log_reg.bin', 'rb') as f_in:
            model = pickle.load(f_in)
            print("Model loaded from local binary file.")
    return model

model = load_model()

app = Flask('Drinking-water-safety-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        if request.content_type == 'application/json':
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            df = get_and_clean_data(data, is_json=True)
        
        elif 'multipart/form-data' in request.content_type:
            if 'file' not in request.files:
                return jsonify({'error': 'No file part in the request'}), 400
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400
            df = get_and_clean_data(file, is_json=False)
        
        else:
            return jsonify({'error': 'Unsupported media type'}), 415

        if 'is_safe' not in df.columns:
            return jsonify({'error': "'is_safe' column is missing from the data."}), 400
        
        X_test = feature_engineering(df, is_prediction=True)
        y_pred = model.predict(X_test).tolist()
        
        response = {
            "predictions": y_pred,
            "message": ["water is safe for drinking" if pred == 1 else "water is not safe for drinking" for pred in y_pred]
        }

        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
