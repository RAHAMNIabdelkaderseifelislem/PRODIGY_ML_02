"""
Utility functions for preprocessing and preparing data for customer segmentation
analysis.

This module provides functions for loading and preprocessing the mall customers
dataset, preparing single customer data for prediction, and saving and loading a
fitted scaler.

"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

class DataPreprocessor:
    """
    Class for preprocessing and preparing data for customer segmentation analysis.
    """
    
    def __init__(self):
        """
        Initialize the preprocessor with a StandardScaler.
        """
        self.scaler = StandardScaler()
        
    def load_data(self, filepath):
        """
        Load and preprocess the mall customers dataset.

        Parameters
        ----------
        filepath : str
            Path to the dataset file.

        Returns
        -------
        df : pd.DataFrame
            Preprocessed dataset.
        """
        try:
            df = pd.read_csv(filepath)
            return df
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def preprocess_data(self, df):
        """
        Preprocess the data for clustering.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to preprocess.

        Returns
        -------
        X_scaled_df : pd.DataFrame
            Scaled dataset.
        X : pd.DataFrame
            Original dataset.
        """
        # Select features for clustering
        features = ['Annual Income (k$)', 'Spending Score (1-100)', 'Age']
        X = df[features].copy()
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to DataFrame with feature names
        X_scaled_df = pd.DataFrame(X_scaled, columns=features)
        
        return X_scaled_df, X
    
    def prepare_single_prediction(self, age, income, spending_score):
        """
        Prepare single customer data for prediction.

        Parameters
        ----------
        age : int
            Age of the customer.
        income : int
            Annual income of the customer.
        spending_score : int
            Spending score of the customer.

        Returns
        -------
        scaled_data : np.ndarray
            Scaled data for prediction.
        """
        data = np.array([[age, income, spending_score]])
        scaled_data = self.scaler.transform(data)
        return scaled_data
    
    def save_scaler(self, filepath):
        """
        Save the fitted scaler.

        Parameters
        ----------
        filepath : str
            Path to save the scaler.
        """
        joblib.dump(self.scaler, filepath)
        
    def load_scaler(self, filepath):
        """
        Load a fitted scaler.

        Parameters
        ----------
        filepath : str
            Path to the saved scaler.
        """
        self.scaler = joblib.load(filepath)
