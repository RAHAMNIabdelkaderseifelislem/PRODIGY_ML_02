import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

class CustomerSegmentation:
    def __init__(self):
        self.model = None
        self.optimal_clusters = None
        
    def find_optimal_clusters(self, X, max_clusters=10):
        """Find optimal number of clusters using elbow method."""
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
        """Train the K-means model with optimal clusters."""
        if self.optimal_clusters is None:
            self.optimal_clusters, _, _ = self.find_optimal_clusters(X)
            
        self.model = KMeans(n_clusters=self.optimal_clusters, random_state=42)
        self.model.fit(X)
        return self.model.labels_
    
    def predict_segment(self, X):
        """Predict the segment for new customer data."""
        if self.model is None:
            raise Exception("Model not trained yet!")
        return self.model.predict(X)
    
    def save_model(self, filepath):
        """Save the trained model."""
        joblib.dump(self.model, filepath)
        
    def load_model(self, filepath):
        """Load a trained model."""
        self.model = joblib.load(filepath)