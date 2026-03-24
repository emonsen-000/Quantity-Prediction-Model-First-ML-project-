# Overview

This project is my first machine learning project as I begin learning machine learning.

Previously, I performed a sales analysis project on an Amazon-style sales dataset.
In this project, I extend that analysis by building a machine learning model to predict product purchase quantity based on several order features.

# The goal of this project is to practice the basic machine learning workflow, including:

- data preparation

- feature engineering

- encoding categorical variables

- training a regression model

- evaluating model performance

- visualizing predictions

This project is part of my early ML learning journey, and I plan to improve it later with more advanced models.

# Dataset Features Used

The model predicts Quantity using the following features:

- UnitPrice
- Discount
- Brand
- Category
- Tax
- ShippingCost
- PaymentMethod
- Country
- Month (extracted from OrderDate)
- Categorical variables were converted into numerical form using Label Encoding. (In Linear Model)
- Categorical variables were converted into a more structural form using One Hot Encoding. (In KNN Regressor Model)

# Machine Learning Model

Model used: 
- Linear Regression.
- KNN Regressor


* The dataset was split using:
Train/Test Split (80% training / 20% testing)

# Model Evaluation

Model performance was evaluated using:

. Mean Absolute Error (MAE)

. Mean Squared Error (MSE)

. R² Score

Example result:

- In Linear Regression Model
. MAE: ~1.02
. MSE: ~1.47
. R² : ~0.26

This means the model explains around 26% of the variation in purchase quantity.

- In KNN Regressor Model
. MAE: 0.44
. MSE: 0.37
. r2: 0.816
This means the model explains around 81.6%% of the variation in purchase quantity.

# Visualization

* In Linear Regression Model
i) A scatter plot is used to compare Actual vs Predicted Quantity.

* In KNN Regressor Model
i) Residual Plot
ii) K2 vs Error Plot

# Project Purpose

This project is intended as a learning exercise while studying machine learning.

It demonstrates my understanding of:

. basic ML workflow

. regression models

. feature preparation

. model evaluation

. result visualization

. Future Improvements

As I continue learning machine learning, I plan to improve this project by trying more advanced models such as:

Random Forest

Gradient Boosting


These models may better capture complex relationships in the data.

# Author

This project is part of my learning journey into data science and machine learning
