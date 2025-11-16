# 🚀 Full-Stack AutoML Classification Tool (Churn Predictor)

## 📝 Overview

A full-stack, automated machine learning web application designed for classification tasks, with a special focus on customer churn prediction.

This tool allows a user to upload any CSV dataset, select a target variable, and initiate a robust, rule-based preprocessing pipeline.

## ✨ Key Features

The backend, built with Python and FastAPI, automatically handles data cleaning by:

* Managing missing values based on data type and percentage.
* Identifying and removing extreme outliers from the training set (using 3-Sigma).
* Dropping non-informative features, such as unique identifiers or high-cardinality categorical columns.

## 🧠 Models Trained

The application then trains, evaluates, and compares four distinct models:

* Logistic Regression
* Random Forest
* XGBoost
* A Multi-Layer Perceptron (Neural Network)

## ⚡ Performance & Prediction

The training process is optimized to leverage GPU acceleration for both XGBoost and TensorFlow (Keras) models if a compatible GPU is available.

The user is presented with the accuracy of each model and can then use a dynamic form to input new data for real-time predictions from all four trained models.
