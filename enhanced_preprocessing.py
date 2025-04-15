import pandas as pd
import numpy as np
import streamlit as st
from data_preprocessing import detect_column_types, preprocess_data
from categorical_handling import (
    identify_categorical_columns,
    encode_categorical_features,
    handle_mixed_data_types,
    handle_high_cardinality_categorical,
    encode_text_features,
    suggest_target_encoding
)

def enhanced_preprocess_data(df, target_column, task_type):

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
    
    # Get target encoding suggestion
    target_encoding_suggestion = suggest_target_encoding(df_processed, target_column)
    
    # Handle target column first
    if task_type == 'Regression':
        # For regression, we need numeric target
        if target_encoding_suggestion['encoding'] == 'none':
            # Already numeric, no transformation needed
            pass
        elif target_encoding_suggestion['encoding'] == 'numeric':
            # Convert to numeric
            df_processed[target_column] = pd.to_numeric(df_processed[target_column])
            preprocessing_info['target_transformation'] = {
                'type': 'numeric_conversion',
                'original_dtype': df[target_column].dtype
            }
        elif target_encoding_suggestion['encoding'] in ['numeric_with_imputation', 'binary', 'multiclass', 'high_cardinality']:
            # Handle mixed data types for regression target
            processed_target, transform_info = handle_mixed_data_types(df_processed, target_column)
            df_processed[target_column] = processed_target
            preprocessing_info['target_transformation'] = transform_info
            
            if transform_info['type'] == 'label_encoding':
                st.warning(f"Target column '{target_column}' contains non-numeric values but was converted to numeric using label encoding for regression.")
                st.info(f"Original values: {transform_info['original_values'][:10]}{'...' if len(transform_info['original_values']) > 10 else ''}")
    else:  # Classification
        # For classification, categorical target is fine but needs encoding
        if target_encoding_suggestion['encoding'] in ['binary', 'multiclass', 'high_cardinality']:
            # Use label encoding for classification target
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df_processed[target_column] = le.fit_transform(df_processed[target_column].astype(str))
            preprocessing_info['target_transformation'] = {
                'type': 'label_encoding',
                'encoder': le,
                'original_values': le.classes_
            }
            st.info(f"Target column '{target_column}' was encoded with {len(le.classes_)} classes.")
    
    # Extract target
    y = df_processed[target_column].values
    
    # Drop target from features
    df_features = df_processed.drop(columns=[target_column])
    
    # Identify categorical columns
    categorical_columns = identify_categorical_columns(df_features)
    
    # Handle high cardinality categorical features
    for col in categorical_columns:
        if col in df_features.columns and df_features[col].nunique() > 20:
            processed_col, transform_info = handle_high_cardinality_categorical(df_features, col)
            df_features[col] = processed_col
            preprocessing_info['encoders'][col] = transform_info
    
    # Encode categorical features
    df_features, cat_encoders = encode_categorical_features(df_features, categorical_columns)
    preprocessing_info['encoders'].update(cat_encoders)
    
    # Identify text columns (long string columns)
    text_columns = [col for col in df_features.columns 
                   if df_features[col].dtype == 'object' 
                   and df_features[col].astype(str).str.len().mean() > 50]
    
    # Encode text features if any
    if text_columns:
        df_features, text_encoders = encode_text_features(df_features, text_columns)
        preprocessing_info['encoders'].update(text_encoders)
    
    # Handle remaining object columns (mixed data types)
    for col in df_features.columns:
        if df_features[col].dtype == 'object':
            processed_col, transform_info = handle_mixed_data_types(df_features, col)
            df_features[col] = processed_col
            preprocessing_info['encoders'][col] = transform_info
    
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
    
    # Get feature names after preprocessing
    feature_names = df_features.columns.tolist()
    
    # Convert to numpy arrays
    X = df_features.values
    
    return X, y, feature_names, preprocessing_info
