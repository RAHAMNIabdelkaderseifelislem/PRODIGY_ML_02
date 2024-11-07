"""
Utility script for customer segmentation analysis.

This script provides a Streamlit-based interface for performing customer
segmentation analysis. It includes functions for loading and preprocessing data,
training a K-means clustering model, predicting segments for new customers, and
visualizing results.

The script is organized into three main sections: data analysis, model training,
and prediction. The data analysis section loads and preprocesses the data,
computes summary statistics, and creates visualizations. The model training
section trains a K-means clustering model on the preprocessed data and saves the
model to a file. The prediction section loads the trained model, prepares input
data for prediction, and displays the predicted segment.

"""

import streamlit as st
import pandas as pd
import plotly.express as px
from src.preprocessing import DataPreprocessor
from src.model import CustomerSegmentation
import numpy as np
import os

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

def main():
    """
    Main entry point for the script.

    This function sets up the Streamlit interface, loads and preprocesses the data,
    trains a K-means clustering model, and predicts segments for new customers.
    """
    st.set_page_config(page_title="Customer Segmentation", layout="wide")
    
    st.title("🛍️ Customer Segmentation Analysis")
    st.sidebar.header("Navigation")
    
    # Initialize objects
    preprocessor = DataPreprocessor()
    segmentation = CustomerSegmentation()
    
    # Check if model exists and load it
    model_path = 'model/customer_segmentation.pkl'
    scaler_path = 'model/scaler.pkl'
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        segmentation.load_model(model_path)
        preprocessor.load_scaler(scaler_path)
        st.sidebar.success("✅ Model loaded successfully!")
    
    # Navigation
    page = st.sidebar.selectbox("Choose a page", ["Data Analysis", "Model Training", "Predict New Customer"])
    
    if page == "Data Analysis":
        show_data_analysis(preprocessor)
    elif page == "Model Training":
        show_model_training(preprocessor, segmentation)
    else:
        show_prediction(preprocessor, segmentation)

def show_data_analysis(preprocessor):
    """
    Displays data analysis results.

    This function loads and preprocesses the data, computes summary statistics,
    and creates visualizations.
    """
    st.header("📊 Data Analysis")
    
    try:
        # Load data
        df = preprocessor.load_data("data/mall_customers.csv")
        
        # Display basic statistics
        st.subheader("Dataset Overview")
        st.write(df.describe())
        
        # Create visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Age Distribution")
            fig_age = px.histogram(df, x="Age", nbins=20)
            st.plotly_chart(fig_age)
        
        with col2:
            st.subheader("Income vs Spending Score")
            fig_scatter = px.scatter(df, x="Annual Income (k$)", 
                                   y="Spending Score (1-100)", 
                                   color="Gender")
            st.plotly_chart(fig_scatter)
    except Exception as e:
        st.error(f"Error in data analysis: {str(e)}")

def show_model_training(preprocessor, segmentation):
    """
    Displays model training results.

    This function loads and preprocesses the data, trains a K-means clustering
    model, and saves the model to a file.
    """
    st.header("🔄 Model Training")
    
    try:
        # Load and preprocess data
        df = preprocessor.load_data("data/mall_customers.csv")
        X_scaled, X = preprocessor.preprocess_data(df)
        
        # Automatically train model if not exists
        model_path = 'model/customer_segmentation.pkl'
        scaler_path = 'model/scaler.pkl'
        
        if not os.path.exists(model_path) or st.button("Retrain Model"):
            with st.spinner("Finding optimal clusters..."):
                n_clusters, inertias, silhouette_scores = segmentation.find_optimal_clusters(X_scaled)
                st.success(f"Optimal number of clusters: {n_clusters}")
                
                # Plot elbow curve
                fig_elbow = px.line(x=range(2, len(inertias) + 2), y=inertias,
                                  labels={"x": "Number of Clusters", "y": "Inertia"})
                st.plotly_chart(fig_elbow)
                
            with st.spinner("Training model..."):
                labels = segmentation.train_model(X_scaled)
                df['Cluster'] = labels
                
                # Plot 3D scatter
                fig_3d = px.scatter_3d(df, x='Annual Income (k$)', 
                                     y='Spending Score (1-100)', 
                                     z='Age',
                                     color='Cluster')
                st.plotly_chart(fig_3d)
                
                # Save model and scaler
                segmentation.save_model(model_path)
                preprocessor.save_scaler(scaler_path)
                st.success("✨ Model trained and saved successfully!")
        else:
            st.info("Model already exists! Click 'Retrain Model' to train a new model.")
            
            # Load existing model predictions
            labels = segmentation.predict_segment(X_scaled)
            df['Cluster'] = labels
            
            # Plot 3D scatter of existing model
            fig_3d = px.scatter_3d(df, x='Annual Income (k$)', 
                                 y='Spending Score (1-100)', 
                                 z='Age',
                                 color='Cluster')
            st.plotly_chart(fig_3d)
    
    except Exception as e:
        st.error(f"Error in model training: {str(e)}")

def show_prediction(preprocessor, segmentation):
    """
    Displays prediction results.

    This function prepares input data for prediction, predicts the segment for
    the input data, and displays the predicted segment.
    """
    st.header("🎯 Predict Customer Segment")
    
    try:
        # Check if model exists
        model_path = 'model/customer_segmentation.pkl'
        if not os.path.exists(model_path):
            st.warning("⚠️ No trained model found! Please go to the Model Training page first.")
            return
        
        # Input form
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            income = st.number_input("Annual Income (k$)", min_value=15, max_value=150, value=50)
            
        with col2:
            spending = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)
            
        if st.button("🔍 Predict Segment"):
            # Prepare and predict
            customer_data = preprocessor.prepare_single_prediction(age, income, spending)
            segment = segmentation.predict_segment(customer_data)[0]
            
            # Display result with enhanced styling
            st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                <h3 style='color: #1f77b4;'>Prediction Results</h3>
                <p style='font-size: 20px;'>Customer belongs to Segment {segment}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Segment characteristics with emojis
            characteristics = {
                0: "💰 Budget Conscious: Lower income, moderate spending",
                1: "⭐ Standard: Average income and spending",
                2: "💎 Premium: High income, high spending",
                3: "🌟 High Value: High income, moderate spending",
                4: "🎯 Careful: High income, low spending"
            }
            
            st.write("### Segment Characteristics:")
            st.info(characteristics.get(segment, "Unknown segment"))
            
            # Add recommendation based on segment
            recommendations = {
                0: "Consider offering budget-friendly promotions and loyalty programs.",
                1: "Target with mid-range products and seasonal offers.",
                2: "Focus on premium products and exclusive services.",
                3: "Provide personalized high-end products with value propositions.",
                4: "Emphasize quality and long-term benefits in marketing."
            }
            
            st.write("### 💡 Marketing Recommendation:")
            st.success(recommendations.get(segment, "No specific recommendation available."))
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")

if __name__ == "__main__":
    main()