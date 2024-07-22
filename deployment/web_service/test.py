import requests
import pandas as pd
import io

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
csv_file = io.StringIO()
df.to_csv(csv_file, index=False)
csv_file.seek(0)  # Move to the start of the file

# Send a POST request to the Flask application
url = 'http://localhost:9696/predict'
files = {'file': ('test.csv', csv_file, 'text/csv')}

response = requests.post(url, files=files)

# Print the response from the server
print(response.status_code)
print(response.json())
