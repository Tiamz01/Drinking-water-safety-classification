## Deploying a model as a web-service

* Creating a virtual environment with Pipenv
* Creating a script for predictiong 
* Putting the script into a Flask app
* Packaging the app to Docker


```bash
docker build -t drinkng-water-safety-clasification-prediction-service:v1 .
```

```bash
docker run -it --rm -p 9696:9696  drinkng-water-safety-clasification-prediction-service:v1
```