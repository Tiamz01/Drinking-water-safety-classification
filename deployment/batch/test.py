import requests


# # Example input data
# data = {
#   "aluminium": [1.65],
#   "ammonia": [9.08],
#   "arsenic": [0.04],
#   "barium": [2.85],
#   "cadmium": [0.007],
#   "chloramine": [0.35],
#   "chromium": [0.83],
#   "copper": [0.17],
#   "flouride": [0.05],
#   "bacteria": [0.20],
#   "viruses": [1],
#   "lead": [0.054],
#   "nitrates": [1.13],
#   "mercury": [0.007],
#   "perchlorate": [37.75],
#   "radium": [6.78],
#   "selenium": [0.08],
#   "silver": [0.34],
#   "uranium": [0.02],
#   "is_safe": [1]
# }

# URL of the file on GitHub
file_url = 'https://raw.githubusercontent.com/Tiamz01/data_repo/main/waterQuality1.csv'

# Fetch the file
response = requests.get(file_url)
if response.status_code == 200:
    files = {'file': ('waterQuality1.csv', response.content)}
    # Upload to Flask endpoint
    response = requests.post('http://localhost:9696/predict', files=files)
    print(response.json())
else:
    print("Failed to fetch file.")


url = 'http://127.0.0.1:9696/predict'
response = requests.post(url, json=data)
print(response.json())
