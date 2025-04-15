"""
Helper functions for model training and evaluation in the Streamlit app
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, mean_absolute_error

# Hyperparameter grids for classification models
CLASSIFICATION_PARAM_GRIDS = {
    "Logistic Regression": {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga']
    },
    "K-Nearest Neighbors": {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance']
    },
    "Support Vector Machine": {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    },
    "Decision Tree": {
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10]
    },
    "Random Forest": {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    "Gradient Boosting": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    "AdaBoost": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    },
    "Naive Bayes": {
        'var_smoothing': [1e-9, 1e-8, 1e-7]
    },
    "Multi-layer Perceptron": {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'alpha': [0.0001, 0.001, 0.01]
    },
    "XGBoost": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    "LightGBM": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
}

# Hyperparameter grids for regression models
REGRESSION_PARAM_GRIDS = {
    "Linear Regression": {},  # No hyperparameters to tune
    "Ridge Regression": {
        'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
    },
    "Lasso Regression": {
        'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
    },
    "Elastic Net": {
        'alpha': [0.01, 0.1, 1.0, 10.0],
        'l1_ratio': [0.1, 0.5, 0.7, 0.9]
    },
    "Support Vector Regression": {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    },
    "K-Nearest Neighbors": {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance']
    },
    "Decision Tree": {
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10]
    },
    "Random Forest": {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    "Gradient Boosting": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    "AdaBoost": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    },
    "Multi-layer Perceptron": {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'alpha': [0.0001, 0.001, 0.01]
    },
    "XGBoost": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    "LightGBM": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
}

def train_classification_models_with_tuning(X_train, X_test, y_train, y_test, selected_models, model_dict, tune_hyperparameters=False):
    """
    Train and evaluate classification models with optional hyperparameter tuning
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    X_test : array-like
        Testing features
    y_train : array-like
        Training target
    y_test : array-like
        Testing target
    selected_models : list
        List of model names to train
    model_dict : dict
        Dictionary mapping model names to model classes
    tune_hyperparameters : bool
        Whether to perform hyperparameter tuning
        
    Returns:
    --------
    dict
        Dictionary containing trained models and their performance metrics
    """
    results = {}
    
    with st.spinner("Training classification models..."):
        progress_bar = st.progress(0)
        total_models = len(selected_models)
        
        for i, model_name in enumerate(selected_models):
            try:
                # Update progress
                progress_bar.progress((i) / total_models)
                
                # Get model class
                model_class = model_dict[model_name]
                
                # Initialize model with default parameters
                if model_name == "Logistic Regression":
                    base_model = model_class(max_iter=1000, random_state=42)
                elif model_name in ["Support Vector Machine", "Decision Tree", "Random Forest", "Gradient Boosting", "AdaBoost", "Multi-layer Perceptron", "XGBoost", "LightGBM"]:
                    base_model = model_class(random_state=42)
                else:
                    base_model = model_class()
                
                # Perform hyperparameter tuning if requested
                if tune_hyperparameters and model_name in CLASSIFICATION_PARAM_GRIDS and CLASSIFICATION_PARAM_GRIDS[model_name]:
                    param_grid = CLASSIFICATION_PARAM_GRIDS[model_name]
                    
                    # Use RandomizedSearchCV for efficiency
                    search = RandomizedSearchCV(
                        base_model, 
                        param_distributions=param_grid,
                        n_iter=5,
                        cv=3,
                        scoring='accuracy',
                        random_state=42,
                        n_jobs=-1 if model_name != "Multi-layer Perceptron" else 1  # MLP doesn't support parallel fitting
                    )
                    
                    # Train model with hyperparameter tuning
                    search.fit(X_train, y_train)
                    model = search.best_estimator_
                    best_params = search.best_params_
                else:
                    # Train model with default parameters
                    model = base_model
                    model.fit(X_train, y_train)
                    best_params = "Default parameters (no tuning)"
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate accuracy
                accuracy = accuracy_score(y_test, y_pred)
                
                # Store results
                results[model_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'best_params': best_params
                }
                
            except Exception as e:
                st.warning(f"Error training {model_name}: {e}")
        
        # Complete progress bar
        progress_bar.progress(1.0)
    
    return results

def train_regression_models_with_tuning(X_train, X_test, y_train, y_test, selected_models, model_dict, tune_hyperparameters=False):
    """
    Train and evaluate regression models with optional hyperparameter tuning
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    X_test : array-like
        Testing features
    y_train : array-like
        Training target
    y_test : array-like
        Testing target
    selected_models : list
        List of model names to train
    model_dict : dict
        Dictionary mapping model names to model classes
    tune_hyperparameters : bool
        Whether to perform hyperparameter tuning
        
    Returns:
    --------
    dict
        Dictionary containing trained models and their performance metrics
    """
    results = {}
    
    with st.spinner("Training regression models..."):
        progress_bar = st.progress(0)
        total_models = len(selected_models)
        
        for i, model_name in enumerate(selected_models):
            try:
                # Update progress
                progress_bar.progress((i) / total_models)
                
                # Get model class
                model_class = model_dict[model_name]
                
                # Initialize model with default parameters
                if model_name in ["Decision Tree", "Random Forest", "Gradient Boosting", "AdaBoost", "Multi-layer Perceptron", "XGBoost", "LightGBM"]:
                    base_model = model_class(random_state=42)
                else:
                    base_model = model_class()
                
                # Perform hyperparameter tuning if requested
                if tune_hyperparameters and model_name in REGRESSION_PARAM_GRIDS and REGRESSION_PARAM_GRIDS[model_name]:
                    param_grid = REGRESSION_PARAM_GRIDS[model_name]
                    
                    # Use RandomizedSearchCV for efficiency
                    search = RandomizedSearchCV(
                        base_model, 
                        param_distributions=param_grid,
                        n_iter=5,
                        cv=3,
                        scoring='r2',
                        random_state=42,
                        n_jobs=-1 if model_name != "Multi-layer Perceptron" else 1  # MLP doesn't support parallel fitting
                    )
                    
                    # Train model with hyperparameter tuning
                    search.fit(X_train, y_train)
                    model = search.best_estimator_
                    best_params = search.best_params_
                else:
                    # Train model with default parameters
                    model = base_model
                    model.fit(X_train, y_train)
                    best_params = "Default parameters (no tuning)"
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Store results
                results[model_name] = {
                    'model': model,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2_score': r2,
                    'best_params': best_params
                }
                
            except Exception as e:
                st.warning(f"Error training {model_name}: {e}")
        
        # Complete progress bar
        progress_bar.progress(1.0)
    
    return results

def create_detailed_visualization(results, task_type):
    """
    Create detailed visualization of model performance
    
    Parameters:
    -----------
    results : dict
        Dictionary containing model results
    task_type : str
        Type of task ('Classification' or 'Regression')
        
    Returns:
    --------
    fig
        Matplotlib figure object
    """
    if not results:
        return None
    
    # Extract model names and performance metrics
    model_names = list(results.keys())
    
    if task_type == 'Classification':
        # For classification, use accuracy
        metrics = [results[model]['accuracy'] for model in model_names]
        metric_name = 'Accuracy'
    else:
        # For regression, use R² score
        metrics = [results[model]['r2_score'] for model in model_names]
        metric_name = 'R² Score'
    
    # Create DataFrame for sorting
    df_results = pd.DataFrame({
        'Model': model_names,
        metric_name: metrics
    })
    
    # Sort by metric in descending order
    df_results = df_results.sort_values(metric_name, ascending=False)
    
    # Create bar chart with custom styling
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use a color palette
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(df_results)))
    
    bars = ax.bar(df_results['Model'], df_results[metric_name], color=colors)
    
    # Add metric values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', rotation=0,
                fontsize=10, fontweight='bold')
    
    # Set title and labels with improved styling
    title = f'{task_type} Model {metric_name} Comparison'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Model', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel(metric_name, fontsize=14, fontweight='bold', labelpad=10)
    
    # Set y-axis limit to accommodate text labels
    y_max = max(df_results[metric_name]) * 1.15
    ax.set_ylim(0, max(y_max, 1.1))
    
    # Improve tick labels
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a horizontal line at y=0 for regression models
    if task_type == 'Regression':
        ax.axhline(y=0, color='red', linestyle='-', alpha=0.3)
    
    # Improve overall appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    return fig
