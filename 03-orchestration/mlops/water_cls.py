from prefect import task, flow
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import mlflow
import os
import pickle

# Define tasks
@task
def get_and_clean_data(input_file):
    data = pd.read_csv(input_file)
    data.replace('#NUM!', np.nan, inplace=True)
    data = data.apply(pd.to_numeric, errors='coerce')
    cleaned_data = data.dropna()
    return cleaned_data

@task
def feature_engineering(cleaned_data):
    X = cleaned_data.drop(columns=['is_safe'])
    y = cleaned_data['is_safe']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

@task
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

@task
def apply_model(X_train, X_test, y_train, y_test, output_file):
    model = load_model()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    df_result = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df_result.to_csv(output_file, index=False)
    
    return df_result

# Define the flow
@flow(name="water_classification")
def water_classifier(input_file: str, output_file: str):
    cleaned_data = get_and_clean_data(input_file)
    X_train, X_test, y_train, y_test = feature_engineering(cleaned_data)
    results = apply_model(X_train, X_test, y_train, y_test, output_file)
    return results

# Run the flow locally (for testing)
if __name__ == "__main__":
    input_file = 'https://raw.githubusercontent.com/Tiamz01/data_repo/main/waterQuality1.csv'
    output_file = 'output/predictions.csv'
    water_classifier.serve(name="Water-quality-classification",
                            tags=["water_quality"],
                            parameters={"input_file": input_file,
                                        "output_file": output_file},
                            interval=5)
