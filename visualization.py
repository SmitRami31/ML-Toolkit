import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

def create_model_comparison_chart(results, task_type):

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

def create_confusion_matrix_plot(y_true, y_pred, class_names=None):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot confusion matrix as heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    
    # Set labels and title
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    
    # Improve tick labels
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.tight_layout()
    
    return fig

def create_roc_curve_plot(y_true, y_pred_proba, model_names):

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ROC curve for each model
    for model_name in model_names:
        if model_name in y_pred_proba:
            try:
                # Get predicted probabilities for positive class
                y_prob = y_pred_proba[model_name]
                
                # Compute ROC curve and ROC area
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                roc_auc = auc(fpr, tpr)
                
                # Plot ROC curve
                ax.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
            except Exception as e:
                st.warning(f"Could not generate ROC curve for {model_name}: {e}")
    
    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    
    # Set labels and title
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold', pad=20)
    
    # Set axis limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    # Add legend
    ax.legend(loc="lower right", fontsize=10)
    
    # Add grid
    ax.grid(linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    return fig

def create_regression_error_plot(y_true, y_pred, model_name):

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot scatter plot of true vs predicted values
    ax.scatter(y_true, y_pred, alpha=0.5)
    
    # Plot perfect prediction line
    ax_min = min(min(y_true), min(y_pred))
    ax_max = max(max(y_true), max(y_pred))
    margin = (ax_max - ax_min) * 0.1
    ax_min -= margin
    ax_max += margin
    
    ax.plot([ax_min, ax_max], [ax_min, ax_max], 'r--', lw=2)
    
    # Set labels and title
    ax.set_xlabel('True Values', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
    ax.set_title(f'True vs Predicted Values - {model_name}', fontsize=14, fontweight='bold', pad=20)
    
    # Set axis limits
    ax.set_xlim([ax_min, ax_max])
    ax.set_ylim([ax_min, ax_max])
    
    # Add grid
    ax.grid(linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    return fig

def create_feature_importance_plot(model, feature_names, model_name, top_n=10):

    try:
        # Check if model has feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            # Get feature importances
            importances = model.feature_importances_
            
            # Create DataFrame for sorting
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            })
            
            # Sort by importance in descending order and take top N
            feature_importance = feature_importance.sort_values('Importance', ascending=False).head(top_n)
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot horizontal bar chart
            bars = ax.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
            
            # Add importance values
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{width:.4f}', va='center', fontsize=10)
            
            # Set labels and title
            ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
            ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
            ax.set_title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold', pad=20)
            
            # Reverse y-axis to show most important features at the top
            ax.invert_yaxis()
            
            # Add grid
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            
            # Improve overall appearance
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            
            return fig
        else:
            return None
    except Exception as e:
        st.warning(f"Could not generate feature importance plot: {e}")
        return None

def create_residual_plot(y_true, y_pred, model_name):

    # Calculate residuals
    residuals = y_true - y_pred
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot scatter plot of predicted values vs residuals
    ax.scatter(y_pred, residuals, alpha=0.5)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    
    # Set labels and title
    ax.set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
    ax.set_ylabel('Residuals', fontsize=12, fontweight='bold')
    ax.set_title(f'Residual Plot - {model_name}', fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(linestyle='--', alpha=0.7)
    
    # Improve overall appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    return fig
