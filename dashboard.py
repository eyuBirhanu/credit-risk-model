import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Bati Bank Risk Scoring", layout="wide")

# --- LOAD ARTIFACTS (Model AND Column Order) ---
@st.cache_resource
def load_artifacts():
    """Loads both the pipeline and the list of feature names."""
    model_path = os.path.join("models", "artifacts", "pipeline.pkl")
    features_path = os.path.join("models", "artifacts", "feature_names.pkl")

    if not os.path.exists(model_path) or not os.path.exists(features_path):
        st.error("Model artifacts not found. Please run `python -m src.train` first.")
        return None, None
        
    pipeline = joblib.load(model_path)
    feature_names = joblib.load(features_path)
    return pipeline, feature_names

pipeline, feature_names = load_artifacts()

# --- TITLE & DESCRIPTION ---
st.title("🏦 Bati Bank Credit Scoring System")
st.markdown("This system predicts the **Credit Risk Probability** of a customer based on their transaction history. High Risk customers are likely to default on a loan.")

# --- SIDEBAR: INPUTS ---
st.sidebar.header("Customer Profile")

def user_input_features():
    # ... (This function remains exactly the same as before) ...
    total_spend = st.sidebar.number_input("Total Spend (Value)", min_value=0.0, value=5000.0)
    avg_trans_val = st.sidebar.number_input("Average Transaction Value", min_value=0.0, value=500.0)
    trans_freq = st.sidebar.number_input("Transaction Frequency", min_value=0, value=10)
    trans_var = st.sidebar.number_input("Transaction Variability (Std Dev)", min_value=0.0, value=50.0)
    recency = st.sidebar.number_input("Recency (Days since last txn)", min_value=0.0, value=2.0)
    product_cat = st.sidebar.selectbox("Product Category", ["Airtime", "Financial Services", "Data", "Utility Bill", "Tv"])
    channel_id = st.sidebar.selectbox("Channel ID", ["Android", "Web", "USSD", "iOS"])
    pricing_strat = st.sidebar.selectbox("Pricing Strategy", ['0', '1', '2', '3', '4']) # Ensure these are strings if model expects strings
    
    data = {
        'Total_Spend': total_spend,
        'Avg_Transaction_Value': avg_trans_val,
        'Transaction_Frequency': trans_freq,
        'Transaction_Variability': trans_var,
        'Recency': recency,
        'ProductCategory': product_cat,
        'ChannelId': channel_id,
        'PricingStrategy': str(pricing_strat) # Force to string to be safe
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# --- MAIN PANEL ---

# --- THE CRITICAL FIX IS HERE ---
# Enforce the column order to match the training data
if feature_names:
    input_df = input_df[feature_names]

st.subheader("Customer Data (Correctly Ordered for Model)")
st.dataframe(input_df)

if st.button("Assess Risk"):
    if pipeline:
        # 1. PREDICT
        probability = pipeline.predict_proba(input_df)[0][1]
        prediction = "High Risk" if probability > 0.5 else "Low Risk"
        
        # ... (The rest of the file is the same as before) ...
        # 2. DISPLAY METRICS
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Risk Prediction", value=prediction, 
                      delta="Approved" if prediction == "Low Risk" else "Rejected",
                      delta_color="inverse")
        with col2:
            st.metric(label="Default Probability", value=f"{probability:.2%}")
            st.progress(float(probability))
            if probability > 0.5:
                st.error("⚠️ This customer shows behavioral patterns associated with default.")
            else:
                st.success("✅ This customer shows healthy transaction behaviors.")

        # 3. EXPLAINABILITY (SHAP)
        st.subheader("🔍 Model Explanation (Why?)")
        with st.spinner("Calculating Feature Importance..."):
            try:
                preprocessor = pipeline.named_steps['preprocessor']
                model = pipeline.named_steps['classifier']
                input_transformed = preprocessor.transform(input_df)
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_transformed)
                
                if isinstance(shap_values, list):
                    shap_values_class1 = shap_values[1]
                else:
                    shap_values_class1 = shap_values
                
                fig, ax = plt.subplots()
                shap.summary_plot(shap_values_class1, input_transformed, plot_type="bar", show=False, max_display=10)
                st.pyplot(fig)
                
                st.info("""
                **How to read this chart:**
                - Longer bars mean that feature had a bigger impact on the decision.
                - This shows exactly which factors drove the Risk Score up or down for this specific customer.
                """)
                
            except Exception as e:
                st.warning(f"Could not generate SHAP explanation: {e}")