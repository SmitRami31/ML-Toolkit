# ML-Toolkit

- **An interactive Streamlit web app that allows users to upload CSV datasets, choose between classification or regression tasks, and compare multiple machine learning models with optional hyperparameter tuning and advanced preprocessing.**

## Features:
- **Upload your own dataset (CSV)**

- **Auto-detect column types and data quality**

- **Choose between Classification and Regression tasks**

- **Select your target column and exclude unnecessary features**

- **Compare performance across multiple ML models:** Logistic Regression, Decision Tree, Random Forest, SVM, MLP, KNN, Gradient Boosting, AdaBoost, XGBoost, LightGBM (if available)

- **Optional hyperparameter tuning**

- **Visual performance metrics:** Classification: Accuracy, Confusion Matrix, ROC Curve, Feature Importance

- **Regression: R², RMSE, MAE, Residual Plots, Error Distributions**

- **Automatically recommends features to drop based on correlation and redundancy**

- **Enhanced preprocessing pipeline for mixed-type columns and categorical encoding**

- **Built-in support for sample datasets (Iris, Diabetes, Boston Housing)**

 ## Dependencies:
 
- **streamlit**

- **pandas, numpy, matplotlib, seaborn**

- **scikit-learn**

- **xgboost (optional)**

- **lightgbm (optional)**

## Custom modules:

- **model_utils.py**

- **visualization.py**

- **data_preprocessing.py**

- **enhanced_preprocessing.py**

- **Make sure these files are present in the same directory.**

## Use Cases:

- **Rapid prototyping of ML workflows**

- **Model benchmarking and comparison**

- **Educational tool for exploring supervised learning**

- **Automated preprocessing insights**
