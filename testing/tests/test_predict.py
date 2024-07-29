import pytest
import pandas as pd
from io import StringIO
from predict import get_and_clean_data, feature_engineering, load_model

# Create a sample CSV file for testing
data = {
    "aluminium": [1.65],
    "ammonia": [9.08],
    "arsenic": [0.04],
    "barium": [2.85],
    "cadmium": [0.007],
    "chloramine": [0.35],
    "chromium": [0.83],
    "copper": [0.17],
    "flouride": [0.05],
    "bacteria": [0.20],
    "viruses": [1],
    "lead": [0.054],
    "nitrates": [1.13],
    "nitrites": [1.13],
    "mercury": [0.007],
    "perchlorate": [37.75],
    "radium": [6.78],
    "selenium": [0.08],
    "silver": [0.34],
    "uranium": [0.02],
    "is_safe": [1]  # Include the target variable for prediction
}

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(data)

# Convert the DataFrame to a CSV file in memory
csv_file = StringIO()
df.to_csv(csv_file, index=False)
csv_file.seek(0)  # Move to the start of the file

def test_get_and_clean_data():
    # Use the in-memory CSV file
    df = get_and_clean_data(csv_file)
    assert not df.empty
    assert 'is_safe' in df.columns

def test_feature_engineering():
    df = get_and_clean_data(csv_file)
    X_test = feature_engineering(df, is_prediction=True)
    assert X_test.shape[1] == df.shape[1] - 1  # Should match the number of features minus the target

def test_load_model():
    model = load_model()
    assert model is not None

def test_predictions():
    # Prepare mock data
    expected_predictions = [0]  # Replace with expected predictions based on your model and data
    
    # Reset csv_file to the start of the file for re-reading
    csv_file.seek(0)
    
    # Call the prediction function
    actual_predictions = model(csv_file)
    
    # Assert actual predictions against expected predictions
    assert actual_predictions == expected_predictions

if __name__ == '__main__':
    pytest.main()
