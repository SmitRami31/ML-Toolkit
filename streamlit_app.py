"""
Interactive Streamlit App for ML Model Comparison
This app allows users to upload CSV files and compare multiple ML models
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import custom modules
from model_utils import (
    train_classification_models_with_tuning,
    train_regression_models_with_tuning,
    CLASSIFICATION_PARAM_GRIDS,
    REGRESSION_PARAM_GRIDS
)
from visualization import (
    create_model_comparison_chart,
    create_confusion_matrix_plot,
    create_roc_curve_plot,
    create_regression_error_plot,
    create_feature_importance_plot,
    create_residual_plot
)
from data_preprocessing import (
    detect_column_types,
    suggest_columns_to_drop
)
from enhanced_preprocessing import enhanced_preprocess_data

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
try:
    from xgboost import XGBRegressor
except ImportError:
    pass
try:
    from lightgbm import LGBMRegressor
except ImportError:
    pass

# Set page configuration
st.set_page_config(
    page_title="ML Model Comparison Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define available models
CLASSIFICATION_MODELS = {
    "Logistic Regression": LogisticRegression,
    "K-Nearest Neighbors": KNeighborsClassifier,
    "Support Vector Machine": SVC,
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
    "Gradient Boosting": GradientBoostingClassifier,
    "AdaBoost": AdaBoostClassifier,
    "Naive Bayes": GaussianNB,
    "Multi-layer Perceptron": MLPClassifier,
}

if XGBOOST_AVAILABLE:
    CLASSIFICATION_MODELS["XGBoost"] = XGBClassifier

if LIGHTGBM_AVAILABLE:
    CLASSIFICATION_MODELS["LightGBM"] = LGBMClassifier

REGRESSION_MODELS = {
    "Linear Regression": LinearRegression,
    "Ridge Regression": Ridge,
    "Lasso Regression": Lasso,
    "Elastic Net": ElasticNet,
    "Support Vector Regression": SVR,
    "K-Nearest Neighbors": KNeighborsRegressor,
    "Decision Tree": DecisionTreeRegressor,
    "Random Forest": RandomForestRegressor,
    "Gradient Boosting": GradientBoostingRegressor,
    "AdaBoost": AdaBoostRegressor,
    "Multi-layer Perceptron": MLPRegressor,
}

if XGBOOST_AVAILABLE:
    REGRESSION_MODELS["XGBoost"] = XGBRegressor

if LIGHTGBM_AVAILABLE:
    REGRESSION_MODELS["LightGBM"] = LGBMRegressor

# Function to load and preprocess data
def load_data(file):
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# Main app function
def main():
    # Add title and description
    st.title("ML Model Comparison Tool")
    st.markdown("""
    This app allows you to upload a CSV file and compare the performance of multiple machine learning models.
    Select classification or regression based on your task, choose the models you want to compare, and see the results!
    """)
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # File upload
    st.sidebar.header("1. Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Load data
        df = load_data(uploaded_file)
        
        if df is not None:
            # Display dataset info
            st.sidebar.success(f"Dataset loaded: {uploaded_file.name}")
            
            # Display data preview in main area
            st.subheader("Data Preview")
            st.write(df.head())
            
            # Display dataset info
            st.subheader("Dataset Information")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Rows: {df.shape[0]}")
                st.write(f"Columns: {df.shape[1]}")
            with col2:
                st.write(f"Missing values: {df.isnull().sum().sum()}")
                st.write(f"Duplicate rows: {df.duplicated().sum()}")
            
            # Detect column types
            column_types = detect_column_types(df)
            
            # Display column types
            st.subheader("Column Type Detection")
            col_type_df = pd.DataFrame({
                'Column Type': [k for k, v in column_types.items() for _ in v],
                'Column Name': [col for k, v in column_types.items() for col in v]
            })
            if not col_type_df.empty:
                st.dataframe(col_type_df)
            
            # Task selection
            st.sidebar.header("2. Select Task")
            task = st.sidebar.radio("Choose task type", ["Classification", "Regression"])
            
            # Target selection
            st.sidebar.header("3. Select Target Variable")
            target_column = st.sidebar.selectbox("Choose target column", df.columns)
            
            if target_column:
                # Show target column information
                st.subheader(f"Target Column: {target_column}")
                
                # Display target column statistics
                if pd.api.types.is_numeric_dtype(df[target_column]):
                    st.write(f"Data type: Numeric")
                    st.write(f"Min: {df[target_column].min()}")
                    st.write(f"Max: {df[target_column].max()}")
                    st.write(f"Mean: {df[target_column].mean()}")
                    st.write(f"Median: {df[target_column].median()}")
                    
                    # Show histogram
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(df[target_column].dropna(), bins=20, alpha=0.7)
                    ax.set_title(f"Distribution of {target_column}")
                    ax.set_xlabel(target_column)
                    ax.set_ylabel("Frequency")
                    st.pyplot(fig)
                else:
                    st.write(f"Data type: Categorical")
                    st.write(f"Number of unique values: {df[target_column].nunique()}")
                    
                    # Show value counts
                    value_counts = df[target_column].value_counts().head(10)
                    st.write("Top 10 values:")
                    st.write(value_counts)
                    
                    # Show bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    value_counts.plot(kind='bar', ax=ax)
                    ax.set_title(f"Value Counts of {target_column}")
                    ax.set_xlabel(target_column)
                    ax.set_ylabel("Count")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                
                # Check for mixed data types in target column
                if df[target_column].dtype == 'object':
                    # Try to convert to numeric
                    numeric_conversion = pd.to_numeric(df[target_column], errors='coerce')
                    
                    # Check if conversion resulted in NaNs
                    if numeric_conversion.isna().any():
                        st.warning(f"⚠️ Target column '{target_column}' contains mixed data types (both numeric and non-numeric values).")
                        st.info("The app will automatically handle this by encoding categorical values.")
                        
                        # Show examples of non-numeric values
                        non_numeric_mask = numeric_conversion.isna()
                        if non_numeric_mask.any():
                            st.write("Examples of non-numeric values:")
                            st.write(df.loc[non_numeric_mask, target_column].head())
                
                # Model selection
                st.sidebar.header("4. Select Models")
                
                if task == "Classification":
                    available_models = list(CLASSIFICATION_MODELS.keys())
                    selected_models = st.sidebar.multiselect(
                        "Choose classification models to compare",
                        available_models,
                        default=available_models[:3]  # Default to first 3 models
                    )
                else:  # Regression
                    available_models = list(REGRESSION_MODELS.keys())
                    selected_models = st.sidebar.multiselect(
                        "Choose regression models to compare",
                        available_models,
                        default=available_models[:3]  # Default to first 3 models
                    )
                
                # Advanced options
                st.sidebar.header("5. Advanced Options")
                show_advanced = st.sidebar.checkbox("Show advanced options", value=False)
                
                if show_advanced:
                    # Feature selection
                    st.sidebar.subheader("Feature Selection")
                    
                    # Get column importance
                    with st.spinner("Analyzing feature importance..."):
                        columns_to_drop = suggest_columns_to_drop(df, target_column)
                    
                    if columns_to_drop:
                        drop_features = st.sidebar.multiselect(
                            "Select features to exclude",
                            df.columns.tolist(),
                            default=columns_to_drop
                        )
                    else:
                        drop_features = st.sidebar.multiselect(
                            "Select features to exclude",
                            df.columns.tolist()
                        )
                else:
                    drop_features = []
                
                # Training settings
                st.sidebar.header("6. Training Settings")
                test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.05)
                random_state = st.sidebar.number_input("Random state", 0, 100, 42)
                tune_hyperparameters = st.sidebar.checkbox("Tune hyperparameters", value=False)
                
                # Train button
                train_button = st.sidebar.button("Train Models")
                
                if train_button:
                    if not selected_models:
                        st.warning("Please select at least one model to train.")
                    else:
                        # Create a copy of the dataframe excluding dropped features
                        if drop_features:
                            df_filtered = df.drop(columns=drop_features)
                            st.info(f"Excluded {len(drop_features)} features: {', '.join(drop_features)}")
                        else:
                            df_filtered = df.copy()
                        
                        # Preprocess data with enhanced preprocessing
                        with st.spinner("Preprocessing data with enhanced categorical handling..."):
                            try:
                                X, y, feature_names, preprocessing_info = enhanced_preprocess_data(df_filtered, target_column, task)
                                
                                # Display preprocessing information
                                st.subheader("Data Preprocessing Summary")
                                
                                # Show target transformation
                                if preprocessing_info['target_transformation']:
                                    target_transform = preprocessing_info['target_transformation']
                                    st.write(f"Target transformation: {target_transform['type']}")
                                    
                                    if target_transform['type'] == 'label_encoding' and 'original_values' in target_transform:
                                        st.write(f"Target classes: {list(target_transform['original_values'])[:10]}{'...' if len(target_transform['original_values']) > 10 else ''}")
                                
                                # Show dropped columns
                                if preprocessing_info['dropped_columns']:
                                    st.write(f"Dropped columns: {preprocessing_info['dropped_columns']}")
                                
                                # Show encoded columns
                                encoded_cols = [col for col, info in preprocessing_info['encoders'].items() 
                                              if info['type'] in ['one_hot', 'label', 'binary']]
                                if encoded_cols:
                                    st.write(f"Encoded columns: {encoded_cols}")
                                
                                # Split data
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=test_size, random_state=random_state
                                )
                                
                                # Scale features
                                scaler = StandardScaler()
                                X_train_scaled = scaler.fit_transform(X_train)
                                X_test_scaled = scaler.transform(X_test)
                                
                                # Train and evaluate models
                                if task == "Classification":
                                    results = train_classification_models_with_tuning(
                                        X_train_scaled, X_test_scaled, y_train, y_test, 
                                        selected_models, CLASSIFICATION_MODELS, tune_hyperparameters
                                    )
                                    
                                    # Visualize results
                                    st.header("Classification Results")
                                    
                                    # Model comparison chart
                                    st.subheader("Model Accuracy Comparison")
                                    fig = create_model_comparison_chart(results, 'Classification')
                                    st.pyplot(fig)
                                    
                                    # Display results in a table
                                    st.subheader("Model Performance Metrics")
                                    
                                    # Extract model names and accuracies
                                    model_names = list(results.keys())
                                    accuracies = [results[model]['accuracy'] for model in model_names]
                                    best_params = [results[model]['best_params'] for model in model_names]
                                    
                                    # Create DataFrame for sorting
                                    df_results = pd.DataFrame({
                                        'Model': model_names,
                                        'Accuracy': accuracies,
                                        'Best Parameters': best_params
                                    })
                                    
                                    # Sort by accuracy in descending order
                                    df_results = df_results.sort_values('Accuracy', ascending=False)
                                    
                                    # Display results
                                    st.dataframe(df_results.style.format({'Accuracy': '{:.4f}'}))
                                    
                                    # Get best model
                                    best_model_name = df_results.iloc[0]['Model']
                                    best_model = results[best_model_name]['model']
                                    
                                    # Additional visualizations for the best model
                                    st.subheader(f"Detailed Analysis for Best Model: {best_model_name}")
                                    
                                    # Feature importance (if available)
                                    if hasattr(best_model, 'feature_importances_'):
                                        st.write("### Feature Importance")
                                        fig = create_feature_importance_plot(best_model, feature_names, best_model_name)
                                        if fig:
                                            st.pyplot(fig)
                                    
                                    # Confusion matrix
                                    st.write("### Confusion Matrix")
                                    y_pred = best_model.predict(X_test_scaled)
                                    
                                    # Get original class names if target was encoded
                                    class_names = None
                                    if preprocessing_info['target_transformation'] and preprocessing_info['target_transformation']['type'] == 'label_encoding':
                                        class_names = preprocessing_info['target_transformation']['original_values']
                                    
                                    fig = create_confusion_matrix_plot(y_test, y_pred, class_names)
                                    st.pyplot(fig)
                                    
                                    # ROC curve (if binary classification and model supports predict_proba)
                                    if len(np.unique(y)) == 2 and hasattr(best_model, 'predict_proba'):
                                        st.write("### ROC Curve")
                                        y_pred_proba = {}
                                        for model_name, result in results.items():
                                            if hasattr(result['model'], 'predict_proba'):
                                                y_pred_proba[model_name] = result['model'].predict_proba(X_test_scaled)[:, 1]
                                        
                                        if y_pred_proba:
                                            fig = create_roc_curve_plot(y_test, y_pred_proba, list(y_pred_proba.keys()))
                                            st.pyplot(fig)
                                    
                                else:  # Regression
                                    results = train_regression_models_with_tuning(
                                        X_train_scaled, X_test_scaled, y_train, y_test, 
                                        selected_models, REGRESSION_MODELS, tune_hyperparameters
                                    )
                                    
                                    # Visualize results
                                    st.header("Regression Results")
                                    
                                    # Model comparison chart
                                    st.subheader("Model R² Score Comparison")
                                    fig = create_model_comparison_chart(results, 'Regression')
                                    st.pyplot(fig)
                                    
                                    # Display results in a table
                                    st.subheader("Model Performance Metrics")
                                    
                                    # Extract model names and metrics
                                    model_names = list(results.keys())
                                    r2_scores = [results[model]['r2_score'] for model in model_names]
                                    rmse_scores = [results[model]['rmse'] for model in model_names]
                                    mae_scores = [results[model]['mae'] for model in model_names]
                                    best_params = [results[model]['best_params'] for model in model_names]
                                    
                                    # Create DataFrame for sorting
                                    df_results = pd.DataFrame({
                                        'Model': model_names,
                                        'R² Score': r2_scores,
                                        'RMSE': rmse_scores,
                                        'MAE': mae_scores,
                                        'Best Parameters': best_params
                                    })
                                    
                                    # Sort by R² score in descending order
                                    df_results = df_results.sort_values('R² Score', ascending=False)
                                    
                                    # Display results
                                    st.dataframe(df_results.style.format({
                                        'R² Score': '{:.4f}', 
                                        'RMSE': '{:.4f}',
                                        'MAE': '{:.4f}'
                                    }))
                                    
                                    # Get best model
                                    best_model_name = df_results.iloc[0]['Model']
                                    best_model = results[best_model_name]['model']
                                    
                                    # Additional visualizations for the best model
                                    st.subheader(f"Detailed Analysis for Best Model: {best_model_name}")
                                    
                                    # Feature importance (if available)
                                    if hasattr(best_model, 'feature_importances_'):
                                        st.write("### Feature Importance")
                                        fig = create_feature_importance_plot(best_model, feature_names, best_model_name)
                                        if fig:
                                            st.pyplot(fig)
                                    
                                    # True vs Predicted plot
                                    st.write("### True vs Predicted Values")
                                    y_pred = best_model.predict(X_test_scaled)
                                    fig = create_regression_error_plot(y_test, y_pred, best_model_name)
                                    st.pyplot(fig)
                                    
                                    # Residual plot
                                    st.write("### Residual Plot")
                                    fig = create_residual_plot(y_test, y_pred, best_model_name)
                                    st.pyplot(fig)
                            
                            except Exception as e:
                                st.error(f"Error during data preprocessing or model training: {str(e)}")
                                st.exception(e)
        
    else:
        # Display instructions when no file is uploaded
        st.info("👈 Please upload a CSV file in the sidebar to get started.")
        
        # Display sample datasets
        st.subheader("Or try a sample dataset:")
        sample_data = st.selectbox(
            "Select sample dataset",
            ["None", "Iris (Classification)", "Boston Housing (Regression)", "Diabetes (Regression)"]
        )
        
        if sample_data != "None":
            if sample_data == "Iris (Classification)":
                from sklearn.datasets import load_iris
                data = load_iris()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df['target'] = data.target
                
            elif sample_data == "Boston Housing (Regression)":
                # Boston Housing dataset is deprecated in scikit-learn, using a workaround
                url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
                df = pd.read_csv(url)
                
            elif sample_data == "Diabetes (Regression)":
                from sklearn.datasets import load_diabetes
                data = load_diabetes()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df['target'] = data.target
            
            # Save to session state
            st.session_state.sample_df = df
            
            # Display data preview
            st.subheader("Data Preview")
            st.write(df.head())
            
            # Use sample button
            if st.button("Use this dataset"):
                st.session_state.df = df
                st.experimental_rerun()

# Run the app
if __name__ == "__main__":
    main()
