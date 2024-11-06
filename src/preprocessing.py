import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def load_data(self, filepath):
        """Load and preprocess the mall customers dataset."""
        try:
            df = pd.read_csv(filepath)
            return df
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def preprocess_data(self, df):
        """Preprocess the data for clustering."""
        # Select features for clustering
        features = ['Annual Income (k$)', 'Spending Score (1-100)', 'Age']
        X = df[features].copy()
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to DataFrame with feature names
        X_scaled_df = pd.DataFrame(X_scaled, columns=features)
        
        return X_scaled_df, X
    
    def prepare_single_prediction(self, age, income, spending_score):
        """Prepare single customer data for prediction."""
        data = np.array([[age, income, spending_score]])
        scaled_data = self.scaler.transform(data)
        return scaled_data
    
    def save_scaler(self, filepath):
        """Save the fitted scaler."""
        joblib.dump(self.scaler, filepath)
        
    def load_scaler(self, filepath):
        """Load a fitted scaler."""
        self.scaler = joblib.load(filepath)