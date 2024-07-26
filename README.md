# End-to-End MLOps Water Quality Classification Project

## Overview
This project focuses on building an end-to-end machine learning operations (MLOps) pipeline for a drink water quality amd safety classification task. The key steps include obtaining and cleaning the data, training models, tracking experiments using MLflow, and deploying the model as a web service using Flask, Docker, and Google Cloud Platform (GCP) services for storage and database management.

## Problem statement
Ensuring access to safe drinking water is a critical public health priority. Contaminated water can lead to severe health issues, and thus, timely and accurate assessment of water quality is essential. Traditional methods for testing water safety are often resource-intensive, requiring specialized equipment and expertise. These methods can be impractical for remote or under-resourced areas.

To address this challenge, we propose leveraging machine learning techniques to classify drinking water safety based on readily available data. By utilizing fast and lightweight ML libraries that can run on CPUs, we aim to create an accessible solution that can be deployed on inexpensive hosting platforms. This approach will enable communities, even those with limited resources, to assess water quality efficiently and take necessary actions to ensure safe drinking water.

The goal of this project is to develop a robust, cost-effective machine learning model for classifying drinking water safety, making it feasible for widespread use in various settings without the need for extensive technical infrastructure.

## 🎯 Goals
This is my MLOps project started during MLOps ZoomCamp'24.

And the main goal is straight-forward: build an end-to-end Machine Learning project:

choose dataset
load & analyze data, preprocess it
train & test ML model
create a model training pipeline
deploy the model (as a web service)
finally monitor performance
And follow MLOps best practices!

Dataset is gotten from kaggle dataset. More about the data can be found in the Data collect and model readme file
Data source:https://www.kaggle.com/datasets/mssmartypants/water-quality/data..

Thanks to MLOps ZoomCamp for the reason to learn many new tools!

## Steps Completed

### Data Acquisition and Cleaning
- **Data Collection**: Obtained the water quality dataset from a reliable source.
- **Data Cleaning**: Performed data preprocessing to handle missing values, outliers, and data normalization to ensure the data is ready for model training.

### Model Training
- **Logistic Regression**: Implemented and trained a logistic regression model to classify water quality.
- **Gradient Boosting Classifier**: Implemented and trained a gradient boosting classifier to improve classification performance.

### Experiment Tracking
- **MLflow Integration**: Utilized MLflow to track experiments, including hyperparameter tuning and model performance metrics.
- **GCP PostgreSQL Database**: Set up a GCP PostgreSQL database as the backend store for MLflow to manage experiment metadata.
- **Google Cloud Storage Bucket**: Configured a Google Cloud Storage bucket to store model artifacts and other related files.

-- Incase the GCP service get disrupted, the code is configure to use the model saved as local binary file.

### Model Deployment
- **Flask Application**: Developed a Flask application to serve the trained model as a web service.
- **Docker**: Containerized the Flask application using Docker to ensure consistency across different deployment environments.
- **Google Cloud Platform (GCP)**: 
  - **Google Cloud Run**: Configured Google Cloud Run to deploy the Dockerized Flask application as a scalable web service.
  - **Google Cloud Storage**: Utilized Google Cloud Storage to load the trained model artifacts in the web service.


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



Monitoring is under development yet (adding Evidently AI).

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