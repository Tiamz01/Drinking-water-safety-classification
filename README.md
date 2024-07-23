# End-to-End MLOps Water Quality Classification Project

## Overview
This project focuses on building an end-to-end machine learning operations (MLOps) pipeline for a water quality classification task. The key steps include obtaining and cleaning the data, training models, tracking experiments using MLflow, and deploying the model as a web service using Flask, Docker, and Google Cloud Platform (GCP) services for storage and database management.

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

### Model Deployment
- **Flask Application**: Developed a Flask application to serve the trained model as a web service.
- **Docker**: Containerized the Flask application using Docker to ensure consistency across different deployment environments.
- **Google Cloud Platform (GCP)**: 
  - **Google Cloud Run**: Configured Google Cloud Run to deploy the Dockerized Flask application as a scalable web service.
  - **Google Cloud Storage**: Utilized Google Cloud Storage to load the trained model artifacts in the web service.

## Technologies and Tools Used
- **Python**: For data processing and model implementation.
- **Scikit-learn**: For machine learning model development.
- **MLflow**: For experiment tracking and model management.
- **Flask**: For developing the web service application.
- **Docker**: For containerizing the web service application.
- **Google Cloud Platform (GCP)**: 
  - **PostgreSQL**: For backend experiment metadata storage.
  - **Google Cloud Storage**: For storing model artifacts.
  - **Google Cloud Run**: For deploying the web service.

This document will be updated as the project progresses and new steps are completed.