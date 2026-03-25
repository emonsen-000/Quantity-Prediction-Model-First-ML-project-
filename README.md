## 📊 Quantity Prediction using Machine Learning
# 📌 Overview
This project is part of my early machine learning journey.
After completing a sales analysis project on an Amazon-style dataset, I extended it by building machine learning models to predict product purchase quantity based on order features.

#The goal is to practice the complete ML workflow:

- Data preparation
- Feature engineering
- Encoding categorical variables
- Model training
- Model evaluation
- Visualization
## 📁 Features Used

The target variable is Quantity, predicted using:

- UnitPrice
- Discount
- Brand
- Category
- Tax
- ShippingCost
- PaymentMethod
- Country
- Month (extracted from OrderDate)

## 🔄 Data Preprocessing
Train/Test Split: 80% / 20%
Feature Scaling applied (for KNN)
# Encoding Strategy:
Label Encoding → used in Linear Regression
One-Hot Encoding → used in KNN Regressor

## ⚠️ Important Learning:
Label Encoding negatively impacted KNN performance because it introduces artificial ordinal relationships.
One-Hot Encoding significantly improved results.

# 🤖 Models Used
- Linear Regression (Baseline Model)
- KNN Regressor (Distance-based model)
- 
## 📊 Model Performance
Model	            |   MAE |	MSE	  |R² Score
Linear Regression | ~1.02	| ~1.47 | ~0.26
KNN Regressor	    |  0.44	| 0.37	| 0.816

## 📉 Visualization Insights
# Scatter Plot Behavior (Important Observation)
The Actual vs Predicted plots appear as lines of dots instead of a continuous spread.

This happens because:
Quantity values are discrete integers (e.g., 1, 2, 3)
KNN predicts average values (e.g., 2.66, 3.33)

As a result:
Points cluster around fixed levels
Creating horizontal “lines” instead of a smooth distribution

👉 This is expected behavior and not a model issue.

## 📈 Visualizations Included
- Actual vs Predicted Scatter Plot (Linear Regression)
- Residual Plot (KNN)
- K vs Error Plot (KNN tuning)

## 🎯 Project Purpose
This project demonstrates my understanding of:

- Machine learning workflow
- Regression techniques
- Feature encoding impact
- Model evaluation metrics
- Data visualization
## 🚀 Future Improvements
- Random Forest
- Gradient Boosting
- Hyperparameter tuning
- Feature importance analysis
## 📌 Conclusion

This project highlights how proper feature encoding and preprocessing can significantly impact model performance, especially for distance-based algorithms like KNN.
