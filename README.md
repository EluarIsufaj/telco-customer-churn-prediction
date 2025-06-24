# Telco Customer Churn Prediction

This project builds a machine learning pipeline to predict customer churn in a telecom dataset. It preprocesses numeric and categorical features, applies logistic regression, and evaluates model performance with accuracy, precision, recall, and F1-score.

## Features
- Data cleaning and preprocessing with scikit-learn pipelines
- Handling categorical variables with OneHotEncoding
- Feature scaling using StandardScaler
- Model training and evaluation with Logistic Regression
- Performance metrics and classification report

## Dataset
The dataset contains customer demographic and account info with a target column indicating churn (Yes/No).

You can download the dataset [here](https://www.kaggle.com/blastchar/telco-customer-churn).

## Usage
1. Clone the repo.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the churn model training script: `python churnModel.py`
4. Check the output metrics to evaluate model performance.

## Requirements
- Python 3.x
- pandas
- scikit-learn
- imblearn

