"""
Utility class for customer segmentation analysis using K-means clustering.

This class provides methods for finding the optimal number of clusters using the
elbow method, training a K-means clustering model, predicting the segment for
new customer data, and saving and loading a trained model.

"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

class CustomerSegmentation:
    """
    Utility class for customer segmentation analysis using K-means clustering.
    """
    
    def __init__(self):
        """
        Initialize the segmentation model with no optimal clusters and no
        trained model.
        """
        self.model = None
        self.optimal_clusters = None
        
    def find_optimal_clusters(self, X, max_clusters=10):
        """
        Find optimal number of clusters using elbow method.
        
        Parameters
        ----------
        X : array-like
            Data to cluster.
        max_clusters : int, optional
            Maximum number of clusters to consider. Default is 10.
        
        Returns
        -------
        optimal_clusters : int
            Optimal number of clusters.
        inertias : list
            List of inertias for each cluster size.
        silhouette_scores : list
            List of silhouette scores for each cluster size.
        """
        inertias = []
        silhouette_scores = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))
            
        # Find elbow point using the derivative method
        diffs = np.diff(inertias)
        elbow_point = np.argmin(diffs) + 2
        
        self.optimal_clusters = elbow_point
        return elbow_point, inertias, silhouette_scores
    
    def train_model(self, X):
        """
        Train the K-means model with optimal clusters.
        
        Parameters
        ----------
        X : array-like
            Data to cluster.
        
        Returns
        -------
        labels : array-like
            Cluster labels for each data point.
        """
        if self.optimal_clusters is None:
            self.optimal_clusters, _, _ = self.find_optimal_clusters(X)
            
        self.model = KMeans(n_clusters=self.optimal_clusters, random_state=42)
        self.model.fit(X)
        return self.model.labels_
    
    def predict_segment(self, X):
        """
        Predict the segment for new customer data.
        
        Parameters
        ----------
        X : array-like
            New customer data to predict.
        
        Returns
        -------
        labels : array-like
            Predicted cluster labels for each new customer.
        """
        if self.model is None:
            raise Exception("Model not trained yet!")
        return self.model.predict(X)
    
    def save_model(self, filepath):
        """
        Save the trained model.
        
        Parameters
        ----------
        filepath : str
            Path to save the model.
        """
        joblib.dump(self.model, filepath)
        
    def load_model(self, filepath):
        """
        Load a trained model.
        
        Parameters
        ----------
        filepath : str
            Path to the saved model.
        """
        self.model = joblib.load(filepath)
