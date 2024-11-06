import streamlit as st
import pandas as pd
import plotly.express as px
from src.preprocessing import DataPreprocessor
from src.model import CustomerSegmentation
import numpy as np

def main():
    st.set_page_config(page_title="Customer Segmentation", layout="wide")
    
    st.title("🛍️ Customer Segmentation Analysis")
    st.sidebar.header("Navigation")
    
    # Initialize objects
    preprocessor = DataPreprocessor()
    segmentation = CustomerSegmentation()
    
    # Navigation
    page = st.sidebar.selectbox("Choose a page", ["Data Analysis", "Model Training", "Predict New Customer"])
    
    if page == "Data Analysis":
        show_data_analysis(preprocessor)
    elif page == "Model Training":
        show_model_training(preprocessor, segmentation)
    else:
        show_prediction(preprocessor, segmentation)

def show_data_analysis(preprocessor):
    st.header("📊 Data Analysis")
    
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

def show_model_training(preprocessor, segmentation):
    st.header("🔄 Model Training")
    
    # Load and preprocess data
    df = preprocessor.load_data("data/mall_customers.csv")
    X_scaled, X = preprocessor.preprocess_data(df)
    
    # Train model
    if st.button("Train Model"):
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
            
            # Save model
            segmentation.save_model('model/customer_segmentation.pkl')
            st.success("Model trained and saved successfully!")

def show_prediction(preprocessor, segmentation):
    st.header("🎯 Predict Customer Segment")
    
    # Load saved model
    try:
        segmentation.load_model('model/customer_segmentation.pkl')
    except:
        st.error("Please train the model first!")
        return
    
    # Input form
    age = st.number_input("Age", min_value=18, max_value=100)
    income = st.number_input("Annual Income (k$)", min_value=15, max_value=150)
    spending = st.number_input("Spending Score (1-100)", min_value=1, max_value=100)
    
    if st.button("Predict Segment"):
        # Prepare and predict
        customer_data = preprocessor.prepare_single_prediction(age, income, spending)
        segment = segmentation.predict_segment(customer_data)[0]
        
        # Display result
        st.success(f"Customer belongs to Segment {segment}")
        st.write("Segment Characteristics:")
        
        characteristics = {
            0: "Budget Conscious: Lower income, moderate spending",
            1: "Standard: Average income and spending",
            2: "Premium: High income, high spending",
            3: "High Value: High income, moderate spending",
            4: "Careful: High income, low spending"
        }
        
        st.write(characteristics.get(segment, "Unknown segment"))

if __name__ == "__main__":
    main()