# 🛍️ Customer Segmentation Analysis - Mall Customer Data 🏪
## Task 2 - Prodigy InfoTech ML Internship

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-orange)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red)](https://streamlit.io/)

### 🎯 Project Overview
Transform raw customer data into actionable insights using K-means clustering! This project helps mall marketing teams identify distinct customer segments for targeted marketing strategies.

### ✨ Features
- 🔍 Advanced data preprocessing and scaling
- 📊 Automatic optimal cluster detection using elbow method
- 🎨 Interactive visualization of customer segments
- 🖥️ User-friendly GUI for real-time customer classification
- 📈 Detailed cluster analysis and insights

### 🛠️ Technology Stack
- Python 3.8+
- scikit-learn
- pandas
- numpy
- streamlit
- plotly

### 🚀 Quick Start
1. Clone the repository
```bash
git clone https://github.com/RAHAMNIabdelkaderseifelislem/PRODIGY_ML_02.git
cd customer-segmentation
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the GUI
```bash
streamlit run app.py
```

### 📁 Project Structure
```
customer-segmentation/
├── data/
│   └── mall_customers.csv
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── visualization.py
├── app.py
├── requirements.txt
└── README.md
```

### 🎓 Model Details
- Algorithm: K-means Clustering
- Features used: Annual Income, Spending Score, Age
- Preprocessing: StandardScaler
- Optimal Clusters: Determined using Elbow Method

### 📊 Sample Visualizations
- 3D scatter plots of customer segments
- Feature importance analysis
- Cluster centroids visualization

### 🤝 Contributing
Feel free to submit issues, fork the repository, and create pull requests for any improvements.

### 📝 License
MIT License

### 🙏 Acknowledgments
- Prodigy InfoTech for the amazing internship opportunity
- Mall customer dataset contributors