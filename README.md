# End-to-End MLOps Water Quality Classification Project

## Overview
This project focuses on building an end-to-end machine learning operations (MLOps) pipeline for a drinking water quality and safety classification task. The key steps include obtaining and cleaning the data, training models, tracking experiments using MLflow, and deploying the model as a web service using Flask, Docker, and Google Cloud Platform (GCP) services for storage and database management.

## Problem statement
Ensuring access to safe drinking water is a critical public health priority. Contaminated water can lead to severe health issues; thus, timely and accurate assessment of water quality is essential. Traditional methods for testing water safety are often resource-intensive, requiring specialized equipment and expertise. These methods can be impractical for remote or under-resourced areas.

To address this challenge, we propose leveraging machine learning techniques to classify drinking water safety based on readily available data. By utilizing fast and lightweight ML libraries that can run on CPUs, we aim to create an accessible solution that can be deployed on inexpensive hosting platforms. This approach will enable communities, even those with limited resources, to assess water quality efficiently and take necessary actions to ensure safe drinking water.

This project aims to develop a robust, cost-effective machine learning model for classifying drinking water safety, making it feasible for widespread use in various settings without the need for extensive technical infrastructure.

## 🎯 Goals
This is my MLOps project that started during MLOps ZoomCamp'24.

And the main goal is straightforward: build an end-to-end Machine Learning project:

choose dataset
load & analyze data, preprocess it
train & test ML model
create a model training pipeline
deploy the model (as a web service)
finally, monitor the performance
And follow MLOps best practices!

The dataset is obtained from the Kaggle dataset. More about the data can be found in the Data collection and model readme file
Data source:https://www.kaggle.com/datasets/mssmartypants/water-quality/data.

Thanks to MLOps ZoomCamp for the reason to learn many new tools!

## Steps Completed

### Data Acquisition and Cleaning
- **Data Collection**: Obtained the water quality dataset from a reliable source.
- **Data Cleaning**: Performed data preprocessing to handle missing values, outliers, and normalization to ensure the data is ready for model training.

### Model Training
- **Logistic Regression**: Implemented and trained a logistic regression model to classify water quality.
- **Gradient Boosting Classifier**: Implemented and trained a gradient boosting classifier to improve classification performance.

After the training, the model with the highest f1 score is logged as the best model.


### Experiment Tracking
- **MLflow Integration**: Utilized MLflow to track experiments, including hyperparameter tuning and model performance metrics.
- **GCP PostgreSQL Database**: Set up a GCP PostgreSQL database as the backend store for MLflow to manage experiment metadata.
- **Google Cloud Storage Bucket**: Configured a Google Cloud Storage bucket to store model artifacts and other related files.

-- In case the GCP service gets disrupted, the code is configured to use the model saved as a local binary file.

**mlflow server setup**

```bash
mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri postgresql://waterDB:padlock02@10.110.160.4:5432/mlflow --default-artifact-root gs://water_quality_model 
```

[View the model tracking and registry here]([http://35.192.179.167:5000/)])
![Experiment tracking](images/exp_tracking.png)
![Model Registry](images/model_registry.png)


### Model Orchestration
- The model training pipeline orchestration was done with Prefect to successfully track the workflow and deployment of the model.

- To view it on the prefect UI, RUN the water_cls.py on the terminal after activating the orchestration environment
```bash
    conda activate orchestration
    pip install -r requirements.txt
    python water_cls.py
    ```
![Orchestration](images/model_registry.png)


### Model Deployment
- **Flask Application**: Developed a Flask application to serve the trained model as a web service.
- **Docker**: Containerized the Flask application using Docker to ensure consistency across different deployment environments.
- **Google Cloud Platform (GCP)**: 
  - **Google Cloud Run**: Configured Google Cloud Run to deploy the Dockerized Flask application as a scalable web service.
  - **Google Cloud Storage**: Utilized Google Cloud Storage to load the trained model artifacts in the web service.



- Navigate to the web_service Directory
    ```bash
    cd web_service
    ```
- Building the docker image
    ```bash
    docker build -t drinkng-water-safety-clasification-prediction-service:v1 .
    ```
- To start the web service, use the following command.
    ```bash
    docker run -it --rm -p 9696:9696  drinkng-water-safety-clasification-prediction-service:v1
    ```

### Setup Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/fashion-mnist-classifier.git
    cd fashion-mnist-classifier
    ```
### Running Unit Tests
Unit tests are located in the testing/tests directory. To run unit tests using pytest:

Navigate to the testing Directory
```bash
    cd testing
```
Then Run the tests
```bash
    pytest tests/test_predict.py
```
### Integration Testing with Docker Compose using Googlecloud
Integration tests ensure that the Docker setup and application work together correctly. This is performed using Docker Compose.

- Navigate to the testing Directory
```bash
    cd testing
```

Build and Run the Integration Tests
```bash
    docker-compose up --build --abort-on-container-exit --exit-code-from test
```

Monitoring is under development yet (adding Evidently AI).

🧰 Tech stack
Python for data processing and model implementation
Scikit-learn for machine learning model development
MLFlow for ML experiment tracking and model management
Prefect for ML workflow orchestration
Docker and docker-compose
For containerizing the web service application.
Google Cloud Platform (GCP): 
  - **PostgreSQL**: For backend experiment metadata storage.
  - **Google Cloud Storage**: For storing model artifacts.
  - **Google Cloud Run**: For deploying the web service.

🚀 Instructions to reproduce
Setup environment
Dataset
Train model
Test prediction service
Deployment and Monitoring
Best practices

Best practices
* [x] Unit tests
* [x] Integration test (== Test prediction service)
* [x] Code formatter (isort, black)
* [x] Makefile
* [x] Pre-commit hooks 

Stay tuned!

Support
🙏 Thank you for your attention and time!

If you experience any issue while following this instruction (or something left unclear), please add it to Issues, I'll be glad to help/fix. And your feedback, questions & suggestions are welcome as well!
Feel free to fork and submit pull requests.
If you find this project helpful, please ⭐️star⭐️ my repo https://github.com/Tiamz01/Drinking-water-safety-classification to help other people discover it 🙏

Made with ❤️ in Nigeria 🇺🇦 Ismail Tiamiyu

This document will be updated as the project progresses and new steps are completed.
