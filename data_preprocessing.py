"""
Data preprocessing utilities for the ML Model Comparison Streamlit App
This module provides functions for automatic feature engineering and data preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import streamlit as st

def detect_column_types(df):
    """
    Automatically detect column types in a DataFrame
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
        
    Returns:
    --------
    dict
        Dictionary with column types classification
    """
    column_types = {
        'numeric': [],
        'categorical': [],
        'datetime': [],
        'text': [],
        'binary': [],
        'id': []
    }
    
    for col in df.columns:
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check if it's likely an ID column
            if col.lower().endswith('id') or col.lower() == 'id':
                column_types['id'].append(col)
            # Check if it's binary
            elif df[col].nunique() == 2:
                column_types['binary'].append(col)
            else:
                column_types['numeric'].append(col)
        
        # Check if column is datetime
        elif pd.api.types.is_datetime64_dtype(df[col]) or pd.to_datetime(df[col], errors='coerce').notna().all():
            column_types['datetime'].append(col)
        
        # Check if it's categorical or text
        else:
            # If few unique values relative to total rows, likely categorical
            if df[col].nunique() < min(20, len(df) * 0.1):
                column_types['categorical'].append(col)
            else:
                column_types['text'].append(col)
    
    return column_types

def preprocess_data(df, target_column, task_type):
    """
    Preprocess data with automatic feature engineering
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    target_column : str
        Name of the target column
    task_type : str
        Type of task ('Classification' or 'Regression')
        
    Returns:
    --------
    tuple
        (X, y, feature_names, preprocessing_info)
    """
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Detect column types
    column_types = detect_column_types(df_processed)
    
    # Store preprocessing information
    preprocessing_info = {
        'column_types': column_types,
        'encoders': {},
        'dropped_columns': [],
        'target_transformation': None
    }
    
    # Handle target column first
    if target_column in column_types['categorical'] or target_column in column_types['text']:
        if task_type == 'Regression':
            # For regression, we need to convert categorical target to numeric
            try:
                # Try to convert to numeric if it contains numeric-like strings
                df_processed[target_column] = pd.to_numeric(df_processed[target_column], errors='coerce')
                
                # Check if conversion resulted in too many NaNs
                if df_processed[target_column].isna().sum() > 0.1 * len(df_processed):
                    # Too many NaNs, use label encoding instead
                    le = LabelEncoder()
                    df_processed[target_column] = le.fit_transform(df[target_column].astype(str))
                    preprocessing_info['target_transformation'] = {
                        'type': 'label_encoding',
                        'encoder': le,
                        'original_values': le.classes_
                    }
                    st.warning(f"Target column '{target_column}' contains non-numeric values but was converted to numeric using label encoding for regression.")
                else:
                    preprocessing_info['target_transformation'] = {
                        'type': 'numeric_conversion',
                        'original_dtype': df[target_column].dtype
                    }
                    st.warning(f"Target column '{target_column}' was converted from string to numeric for regression.")
            except:
                # If conversion fails, use label encoding
                le = LabelEncoder()
                df_processed[target_column] = le.fit_transform(df[target_column].astype(str))
                preprocessing_info['target_transformation'] = {
                    'type': 'label_encoding',
                    'encoder': le,
                    'original_values': le.classes_
                }
                st.warning(f"Target column '{target_column}' contains non-numeric values but was converted to numeric using label encoding for regression.")
        else:  # Classification
            # For classification, use label encoding for the target
            le = LabelEncoder()
            df_processed[target_column] = le.fit_transform(df[target_column].astype(str))
            preprocessing_info['target_transformation'] = {
                'type': 'label_encoding',
                'encoder': le,
                'original_values': le.classes_
            }
            st.info(f"Target column '{target_column}' was encoded with {len(le.classes_)} classes: {list(le.classes_)}")
    elif target_column in column_types['numeric'] or target_column in column_types['binary']:
        # Check if numeric column contains any non-numeric strings
        if df[target_column].dtype == 'object':
            try:
                # Try to convert to numeric
                df_processed[target_column] = pd.to_numeric(df_processed[target_column], errors='coerce')
                
                # Check if conversion resulted in NaNs
                if df_processed[target_column].isna().sum() > 0:
                    # Handle NaNs by using label encoding instead
                    le = LabelEncoder()
                    df_processed[target_column] = le.fit_transform(df[target_column].astype(str))
                    preprocessing_info['target_transformation'] = {
                        'type': 'label_encoding',
                        'encoder': le,
                        'original_values': le.classes_
                    }
                    st.warning(f"Target column '{target_column}' contains mixed numeric and non-numeric values. Used label encoding.")
                else:
                    preprocessing_info['target_transformation'] = {
                        'type': 'numeric_conversion',
                        'original_dtype': df[target_column].dtype
                    }
            except:
                # If conversion fails completely, use label encoding
                le = LabelEncoder()
                df_processed[target_column] = le.fit_transform(df[target_column].astype(str))
                preprocessing_info['target_transformation'] = {
                    'type': 'label_encoding',
                    'encoder': le,
                    'original_values': le.classes_
                }
                st.warning(f"Target column '{target_column}' contains non-numeric values but was converted to numeric using label encoding.")
    
    # Extract target
    y = df_processed[target_column].values
    
    # Drop target from features
    df_features = df_processed.drop(columns=[target_column])
    
    # Process feature columns
    for col_type, columns in column_types.items():
        for col in columns:
            if col == target_column or col in preprocessing_info['dropped_columns']:
                continue
                
            # Handle different column types
            if col_type == 'numeric':
                # Keep numeric columns as is
                pass
                
            elif col_type == 'categorical':
                # Handle categorical columns with one-hot encoding for few categories
                if df[col].nunique() <= 10:  # One-hot encode if 10 or fewer categories
                    # Create one-hot encoder
                    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded = ohe.fit_transform(df_features[[col]])
                    
                    # Create new column names
                    encoded_cols = [f"{col}_{cat}" for cat in ohe.categories_[0]]
                    
                    # Add encoded columns to dataframe
                    for i, encoded_col in enumerate(encoded_cols):
                        df_features[encoded_col] = encoded[:, i]
                    
                    # Drop original column
                    df_features = df_features.drop(columns=[col])
                    
                    # Store encoder
                    preprocessing_info['encoders'][col] = {
                        'type': 'one_hot',
                        'encoder': ohe,
                        'encoded_columns': encoded_cols
                    }
                else:
                    # Use label encoding for categorical columns with many categories
                    le = LabelEncoder()
                    df_features[col] = le.fit_transform(df_features[col].astype(str))
                    
                    # Store encoder
                    preprocessing_info['encoders'][col] = {
                        'type': 'label',
                        'encoder': le
                    }
            
            elif col_type == 'datetime':
                # Extract useful features from datetime
                df_features[f"{col}_year"] = pd.to_datetime(df[col], errors='coerce').dt.year
                df_features[f"{col}_month"] = pd.to_datetime(df[col], errors='coerce').dt.month
                df_features[f"{col}_day"] = pd.to_datetime(df[col], errors='coerce').dt.day
                df_features[f"{col}_dayofweek"] = pd.to_datetime(df[col], errors='coerce').dt.dayofweek
                
                # Drop original column
                df_features = df_features.drop(columns=[col])
                
                # Store transformation
                preprocessing_info['encoders'][col] = {
                    'type': 'datetime_features',
                    'features': ['year', 'month', 'day', 'dayofweek']
                }
            
            elif col_type == 'text':
                # For text columns, use label encoding
                le = LabelEncoder()
                df_features[col] = le.fit_transform(df_features[col].astype(str))
                
                # Store encoder
                preprocessing_info['encoders'][col] = {
                    'type': 'label',
                    'encoder': le
                }
            
            elif col_type == 'binary':
                # Ensure binary columns are 0/1
                le = LabelEncoder()
                df_features[col] = le.fit_transform(df_features[col].astype(str))
                
                # Store encoder
                preprocessing_info['encoders'][col] = {
                    'type': 'binary',
                    'encoder': le
                }
            
            elif col_type == 'id':
                # Drop ID columns
                df_features = df_features.drop(columns=[col])
                preprocessing_info['dropped_columns'].append(col)
    
    # Handle missing values
    for col in df_features.columns:
        if df_features[col].isna().any():
            if pd.api.types.is_numeric_dtype(df_features[col]):
                # Fill numeric columns with median
                median_value = df_features[col].median()
                df_features[col] = df_features[col].fillna(median_value)
            else:
                # Fill categorical columns with mode
                mode_value = df_features[col].mode()[0]
                df_features[col] = df_features[col].fillna(mode_value)
    
    # Convert all remaining object columns to numeric
    for col in df_features.columns:
        if df_features[col].dtype == 'object':
            try:
                df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
                df_features[col] = df_features[col].fillna(df_features[col].median())
            except:
                # If conversion fails, drop the column
                df_features = df_features.drop(columns=[col])
                preprocessing_info['dropped_columns'].append(col)
    
    # Get feature names after preprocessing
    feature_names = df_features.columns.tolist()
    
    # Convert to numpy arrays
    X = df_features.values
    
    return X, y, feature_names, preprocessing_info

def get_column_importance(df, target_column):
    """
    Calculate importance of each column for predicting the target
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    target_column : str
        Name of the target column
        
    Returns:
    --------
    dict
        Dictionary with column importance scores
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Handle categorical columns
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))
    
    # Handle missing values
    df_copy = df_copy.fillna(df_copy.median(numeric_only=True))
    
    # Separate features and target
    X = df_copy.drop(columns=[target_column])
    y = df_copy[target_column]
    
    # Check if target is categorical
    if pd.api.types.is_numeric_dtype(y) and len(y.unique()) > 10:
        # Regression task
        model = RandomForestRegressor(n_estimators=50, random_state=42)
    else:
        # Classification task
        model = RandomForestClassifier(n_estimators=50, random_state=42)
    
    try:
        # Train model
        model.fit(X, y)
        
        # Get feature importance
        importance = model.feature_importances_
        
        # Create dictionary of column importance
        column_importance = {col: imp for col, imp in zip(X.columns, importance)}
        
        return column_importance
    except:
        # If model training fails, return None
        return None

def suggest_columns_to_drop(df, target_column, threshold=0.01):
    """
    Suggest columns to drop based on importance
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    target_column : str
        Name of the target column
    threshold : float
        Importance threshold below which columns are suggested to drop
        
    Returns:
    --------
    list
        List of columns suggested to drop
    """
    # Get column importance
    column_importance = get_column_importance(df, target_column)
    
    if column_importance is None:
        return []
    
    # Find columns with importance below threshold
    columns_to_drop = [col for col, imp in column_importance.items() if imp < threshold]
    
    return columns_to_drop
