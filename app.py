import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import time

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #3366ff;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #5c5c8a;
        margin-bottom: 1rem;
        text-align: center;
    }
    .card {
        border-radius: 10px;
        padding: 1.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .prediction-card {
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
        text-align: center;
    }
    .high-risk {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .low-risk {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    .stProgress > div > div > div > div {
        background-color: #3366ff;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model
model = tf.keras.models.load_model('ANNmodel.keras')

# Load the encoders and scaler
with open('gender_label.pkl', 'rb') as file:
    gender_encoder = pickle.load(file)

with open('geo_encoder.pkl', 'rb') as file:
    geo_encoder = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# Main App Header
st.markdown('<h1 class="main-header">Customer Churn Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict if a customer is likely to leave your service</p>', unsafe_allow_html=True)

# Add a brief explanation
with st.expander("ℹ️ About this app", expanded=False):
    st.write("""
    This app predicts the probability of a customer churning (leaving) based on various factors.
    Enter the customer details below to get a prediction.
    
    The model was trained on historical customer data using an Artificial Neural Network.
    """.strip())


# Create a two-column layout
col1, col2 = st.columns(2)

# Customer Demographics Card
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📋 Customer Demographics")
    geography = st.selectbox('Geography', geo_encoder.categories_[0], help="Customer's country of residence")
    gender = st.selectbox('Gender', gender_encoder.classes_, help="Customer's gender")
    age = st.slider('Age', 18, 92, 35, help="Customer's age in years")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Account Details Card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💰 Account Details")
    balance = st.number_input('Balance', min_value=0.0, value=76000.0, step=1000.0, help="Customer's account balance")
    tenure = st.slider('Tenure', 0, 10, 5, help="Number of years the customer has been with the bank")
    num_of_products = st.slider('Number of Products', 1, 4, 1, help="Number of bank products the customer uses")
    st.markdown('</div>', unsafe_allow_html=True)

# Financial Information Card
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💳 Financial Information")
    credit_score = st.number_input('Credit Score', min_value=300, max_value=900, value=650, step=1, help="Customer's credit score")
    estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=100000.0, step=1000.0, help="Customer's estimated annual salary")
    has_cr_card = st.selectbox('Has Credit Card', ["Yes", "No"], help="Whether the customer has a credit card")
    is_active_member = st.selectbox('Is Active Member', ["Yes", "No"], help="Whether the customer is an active member")
    st.markdown('</div>', unsafe_allow_html=True)

# Convert Yes/No to 1/0
has_cr_card_value = 1 if has_cr_card == "Yes" else 0
is_active_member_value = 1 if is_active_member == "Yes" else 0

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card_value],
    'IsActiveMember': [is_active_member_value],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = geo_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_encoder.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Add a predict button with animation
if st.button('🔮 Predict Churn Probability', use_container_width=True):
    with st.spinner('Analyzing customer data...'):
        # Add a small delay for effect
        time.sleep(1)
        
        # Predict churn
        prediction = model.predict(input_data_scaled)
        prediction_proba = prediction[0][0]
        
        # Display prediction with appropriate styling
        if prediction_proba > 0.5:
            risk_class = "high-risk"
            emoji = "⚠️"
            message = "High Risk of Churn"
        else:
            risk_class = "low-risk"
            emoji = "✅"
            message = "Low Risk of Churn"
        
        # Display the prediction card with appropriate styling
        st.markdown(f'<div class="prediction-card {risk_class}">', unsafe_allow_html=True)
        st.markdown(f"<h2>{emoji} {message}</h2>", unsafe_allow_html=True)
        
        # Progress bar for visualization
        st.progress(float(prediction_proba))
        st.markdown(f"<h3>Churn Probability: {prediction_proba:.2%}</h3>", unsafe_allow_html=True)
        
        # Add interpretation
        if prediction_proba > 0.5:
            st.markdown("<p>This customer is likely to leave your service. Consider proactive retention strategies.</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p>This customer is likely to stay. Continue providing excellent service.</p>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Feature importance visualization
        st.subheader("Key Factors Analysis")
        
        # Create a simple visualization of the input factors
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Define factors that typically influence churn
        factors = ['Age', 'Credit Score', 'Balance', 'Tenure', 'Products', 'Active Member']
        values = [age/92, credit_score/900, balance/200000, tenure/10, num_of_products/4, is_active_member_value]
        
        # Create horizontal bar chart
        bars = ax.barh(factors, values, color=['#3366ff', '#ff6b6b', '#4ecdc4', '#ffe66d', '#6b5b95', '#88d8b0'])
        
        # Add labels and styling
        ax.set_xlim(0, 1)
        ax.set_xlabel('Normalized Value')
        ax.set_title('Customer Profile Factors')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                    ha='left', va='center')
        
        st.pyplot(fig)
        
        # Add recommendations based on prediction
        st.subheader("Recommendations")
        if prediction_proba > 0.7:
            st.error("Urgent: This customer has a very high risk of churning. Immediate intervention recommended.")
            st.markdown("- Offer a personalized retention package")
            st.markdown("- Schedule a follow-up call with a customer service representative")
            st.markdown("- Consider a loyalty discount or upgrade")
        elif prediction_proba > 0.5:
            st.warning("This customer has a moderate risk of churning. Proactive measures recommended.")
            st.markdown("- Send a customer satisfaction survey")
            st.markdown("- Provide information about new features or services")
            st.markdown("- Consider a small loyalty reward")
        else:
            st.success("This customer is likely to stay. Maintain the relationship.")
            st.markdown("- Continue regular engagement")
            st.markdown("- Consider cross-selling opportunities")
            st.markdown("- Include in regular customer appreciation programs")

# Add a footer with author information
st.markdown("""<hr style='margin-top: 2rem; margin-bottom: 1rem;'>""", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem 0; font-size: 0.8rem;">
    <p>Built by <b>Jaganmohan Reddy Peravali</b></p>
    <p>
        <a href="mailto:peravali810@gmail.com" style="color: #3366ff; text-decoration: none; margin: 0 10px;">peravali810@gmail.com</a> | 
        <a href="https://github.com/pjmreddy" target="_blank" style="color: #3366ff; text-decoration: none; margin: 0 10px;">GitHub</a> | 
        <a href="https://www.linkedin.com/in/pjmreddy14/" target="_blank" style="color: #3366ff; text-decoration: none; margin: 0 10px;">LinkedIn</a>
    </p>
</div>
""", unsafe_allow_html=True)
