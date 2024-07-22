 # End-to-End MLOps Water Quality Classification Project

## Overview
This project focuses on building an end-to-end machine learning operations (MLOps) pipeline for a water quality classification task. The key steps include obtaining and cleaning the data, training models, and tracking experiments using MLflow, with Google Cloud Platform (GCP) services for storage and database management.

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

## Technologies and Tools Used
- **Python**: For data processing and model implementation.
- **Scikit-learn**: For machine learning model development.
- **MLflow**: For experiment tracking and model management.
- **Google Cloud Platform (GCP)**: 
  - **PostgreSQL**: For backend experiment metadata storage.
  - **Google Cloud Storage**: For storing model artifacts.

