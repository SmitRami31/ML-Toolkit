import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA

def identify_categorical_columns(df):

    categorical_columns = []
    
    for col in df.columns:
        # Check if column is already categorical dtype
        if pd.api.types.is_categorical_dtype(df[col]):
            categorical_columns.append(col)
            continue
            
        # Check if column is object dtype
        if df[col].dtype == 'object':
            # Check if it's likely a text column (long strings)
            if df[col].str.len().mean() > 50:  # Long text
                continue
                
            categorical_columns.append(col)
            continue
            
        # Check if numeric column with few unique values
        if pd.api.types.is_numeric_dtype(df[col]):
            n_unique = df[col].nunique()
            if n_unique < 20 and n_unique / len(df) < 0.05:
                categorical_columns.append(col)
                continue
    
    return categorical_columns

def encode_categorical_features(df, categorical_columns=None, encoding_method='auto', max_categories=10):

    # Make a copy to avoid modifying the original
    df_encoded = df.copy()
    encoders_dict = {}
    
    # Identify categorical columns if not provided
    if categorical_columns is None:
        categorical_columns = identify_categorical_columns(df)
    
    # Process each categorical column
    for col in categorical_columns:
        if col not in df_encoded.columns:
            continue
            
        # Determine encoding method if auto
        method = encoding_method
        if method == 'auto':
            n_unique = df_encoded[col].nunique()
            if n_unique <= max_categories:
                method = 'onehot'
            else:
                method = 'label'
        
        # Apply encoding
        if method == 'label':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders_dict[col] = {'type': 'label', 'encoder': le}
            
        elif method == 'onehot':
            # Create one-hot encoder
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded = ohe.fit_transform(df_encoded[[col]])
            
            # Create new column names
            encoded_cols = [f"{col}_{cat}" for cat in ohe.categories_[0]]
            
            # Add encoded columns to dataframe
            for i, encoded_col in enumerate(encoded_cols):
                df_encoded[encoded_col] = encoded[:, i]
            
            # Drop original column
            df_encoded = df_encoded.drop(columns=[col])
            
            # Store encoder
            encoders_dict[col] = {'type': 'onehot', 'encoder': ohe, 'encoded_columns': encoded_cols}
            
        elif method == 'target':
            # Target encoding requires target column, so skip for now
            # This would be implemented in a separate function
            pass
    
    return df_encoded, encoders_dict

def handle_mixed_data_types(df, column):

    # Make a copy to avoid modifying the original
    series = df[column].copy()
    
    # Check if already numeric
    if pd.api.types.is_numeric_dtype(series):
        return series, {'type': 'already_numeric'}
    
    # Try to convert to numeric
    numeric_series = pd.to_numeric(series, errors='coerce')
    
    # Check if conversion resulted in NaNs
    na_count = numeric_series.isna().sum()
    na_percentage = na_count / len(numeric_series)
    
    if na_count == 0:
        # Perfect conversion to numeric
        return numeric_series, {'type': 'numeric_conversion'}
    elif na_percentage < 0.1:
        # Few NaNs, can be imputed
        median_value = numeric_series.median()
        numeric_series = numeric_series.fillna(median_value)
        
        # Store non-numeric values
        non_numeric_mask = pd.isna(numeric_series)
        non_numeric_values = series[non_numeric_mask].unique().tolist()
        
        return numeric_series, {
            'type': 'numeric_with_imputation',
            'imputed_value': median_value,
            'non_numeric_values': non_numeric_values
        }
    else:
        # Too many NaNs, use label encoding
        le = LabelEncoder()
        encoded_series = le.fit_transform(series.astype(str))
        
        return encoded_series, {
            'type': 'label_encoding',
            'encoder': le,
            'original_values': le.classes_
        }

def handle_high_cardinality_categorical(df, column, max_categories=20, method='frequency'):

    # Make a copy to avoid modifying the original
    series = df[column].copy()
    
    if method == 'frequency':
        # Keep top categories by frequency and group others
        value_counts = series.value_counts()
        top_categories = value_counts.nlargest(max_categories).index.tolist()
        
        # Replace less frequent categories with 'Other'
        processed_series = series.copy()
        processed_series[~processed_series.isin(top_categories)] = 'Other'
        
        return processed_series, {
            'type': 'frequency_encoding',
            'top_categories': top_categories
        }
        
    elif method == 'clustering':
        # Convert categories to numeric using label encoding
        le = LabelEncoder()
        numeric_representation = le.fit_transform(series.astype(str)).reshape(-1, 1)
        
        # Cluster categories
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=max_categories, random_state=42)
        clusters = kmeans.fit_predict(numeric_representation)
        
        # Create mapping from original categories to clusters
        category_to_cluster = {category: cluster for category, cluster in zip(le.classes_, clusters)}
        
        # Apply mapping
        processed_series = series.map(lambda x: f"Cluster_{category_to_cluster.get(x, 0)}")
        
        return processed_series, {
            'type': 'clustering',
            'category_to_cluster': category_to_cluster
        }
        
    elif method == 'hash':
        # Hash encoding
        processed_series = series.apply(lambda x: hash(str(x)) % max_categories)
        
        return processed_series, {
            'type': 'hash_encoding',
            'n_categories': max_categories
        }
    
    # Default fallback to label encoding
    le = LabelEncoder()
    encoded_series = le.fit_transform(series.astype(str))
    
    return encoded_series, {
        'type': 'label_encoding',
        'encoder': le
    }

def encode_text_features(df, text_columns, max_features=100):

    # Make a copy to avoid modifying the original
    df_encoded = df.copy()
    encoders_dict = {}
    
    for col in text_columns:
        if col not in df_encoded.columns:
            continue
            
        # Create bag of words representation
        vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
        bow_matrix = vectorizer.fit_transform(df_encoded[col].fillna('').astype(str))
        
        # If many features, use PCA to reduce dimensionality
        if bow_matrix.shape[1] > 10:
            n_components = min(10, bow_matrix.shape[1])
            pca = PCA(n_components=n_components)
            reduced_features = pca.fit_transform(bow_matrix.toarray())
            
            # Add PCA components as new columns
            for i in range(n_components):
                df_encoded[f"{col}_pca_{i}"] = reduced_features[:, i]
                
            # Store encoders
            encoders_dict[col] = {
                'type': 'text_pca',
                'vectorizer': vectorizer,
                'pca': pca
            }
        else:
            # Add bag of words features as new columns
            feature_names = vectorizer.get_feature_names_out()
            bow_df = pd.DataFrame(bow_matrix.toarray(), columns=[f"{col}_bow_{f}" for f in feature_names])
            
            # Concatenate with original dataframe
            df_encoded = pd.concat([df_encoded, bow_df], axis=1)
            
            # Store encoder
            encoders_dict[col] = {
                'type': 'text_bow',
                'vectorizer': vectorizer
            }
        
        # Drop original column
        df_encoded = df_encoded.drop(columns=[col])
    
    return df_encoded, encoders_dict

def suggest_target_encoding(df, target_column, categorical_columns=None):

    # Get target series
    target = df[target_column]
    
    # Check if already numeric
    if pd.api.types.is_numeric_dtype(target):
        return {
            'encoding': 'none',
            'reason': 'Target is already numeric',
            'unique_values': target.nunique()
        }
    
    # Try to convert to numeric
    numeric_target = pd.to_numeric(target, errors='coerce')
    
    # Check if conversion resulted in NaNs
    na_count = numeric_target.isna().sum()
    na_percentage = na_count / len(numeric_target)
    
    if na_count == 0:
        # Perfect conversion to numeric
        return {
            'encoding': 'numeric',
            'reason': 'Target can be directly converted to numeric',
            'unique_values': numeric_target.nunique()
        }
    elif na_percentage < 0.1:
        # Few NaNs, can be imputed
        return {
            'encoding': 'numeric_with_imputation',
            'reason': f'Target is mostly numeric with {na_count} non-numeric values',
            'non_numeric_examples': target[numeric_target.isna()].unique().tolist()[:5],
            'unique_values': target.nunique()
        }
    else:
        # Categorical target
        n_unique = target.nunique()
        
        if n_unique == 2:
            return {
                'encoding': 'binary',
                'reason': 'Target is binary categorical',
                'categories': target.unique().tolist()
            }
        elif n_unique <= 10:
            return {
                'encoding': 'multiclass',
                'reason': 'Target is multiclass categorical with few classes',
                'n_classes': n_unique,
                'categories': target.unique().tolist()
            }
        else:
            return {
                'encoding': 'high_cardinality',
                'reason': 'Target is categorical with many classes',
                'n_classes': n_unique,
                'top_categories': target.value_counts().nlargest(5).index.tolist()
            }
