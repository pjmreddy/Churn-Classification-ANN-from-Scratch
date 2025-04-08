<div align="center">

# Churn Classification ANN from Scratch

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://churn-classification-ann-from-scratch.streamlit.app/)

An Artificial neural network implementation for predicting customer churn built from scratch

</div>

## 📋 Project Overview

This project implements an Artificial Neural Network (ANN) from scratch to predict customer churn for a bank. The model analyzes various customer attributes to determine the likelihood of a customer leaving the bank's services.

### Key Features

- Custom ANN implementation using TensorFlow/Keras
- Interactive web application built with Streamlit
- Comprehensive data preprocessing pipeline
- Visualization of prediction results
- Model performance evaluation

## 🧠 Neural Network Architecture

The ANN model consists of:
- Input layer with 12 features
- Hidden layers with ReLU activation
- Output layer with sigmoid activation for binary classification
- Early stopping to prevent overfitting
- TensorBoard integration for model monitoring

## 📊 Dataset

The model is trained on a bank customer dataset with the following features:
- Customer demographics (age, gender, geography)
- Account information (balance, tenure, number of products)
- Engagement metrics (credit score, active membership status)
- Target variable: Exited (whether the customer left the bank)

## 🚀 Web Application

The project includes a Streamlit web application that allows users to:
- Input customer information through an intuitive interface
- Receive real-time churn predictions
- Visualize key factors affecting the prediction
- Understand the likelihood of customer churn

## 🛠️ Technologies Used

- Python
- TensorFlow
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit

## 🔧 Installation & Usage

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## 🌐 Live Demo

Explore the live application: [Churn Classification ANN](https://churn-classification-ann-from-scratch.streamlit.app/)

## 📝 Project Structure

- `ANN_from_Scratch.ipynb`: Jupyter notebook with model development
- `app.py`: Streamlit application code
- `ANNmodel.keras`: Saved neural network model
- `requirements.txt`: Project dependencies
- `*.pkl`: Saved encoders and scaler for preprocessing

## 📈 Future Improvements

- Hyperparameter tuning for better model performance
- Additional feature engineering
- Model comparison with other algorithms
- Enhanced visualization capabilities

<div align="center">

---

Developed with ❤️ by [Jagan Reddy](mailto:peravali810@gmail.com).

</div>