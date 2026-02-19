# src/train.py
import pandas as pd
import mlflow
import mlflow.sklearn
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.pipeline import Pipeline
# Import our new modules
from src.features import CreditRiskPreprocessor, aggregate_transactions
from src.utils import save_object
from src.data_processing import create_rfm_risk_label # We keep your RFM logic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CONFIG
RAW_DATA_PATH = 'data/raw/data.csv'
ARTIFACT_PATH = 'models/artifacts'
EXPERIMENT_NAME = "Bati_Bank_Credit_Score_v2"

def main():
    # 1. Load and Aggregate
    logger.info("Loading raw data...")
    df_raw = pd.read_csv(RAW_DATA_PATH)
    df_raw['TransactionStartTime'] = pd.to_datetime(df_raw['TransactionStartTime'])
    
    logger.info("Aggregating to Customer Level...")
    df_cust = aggregate_transactions(df_raw)
    
    # 2. Create Target (RFM Proxy)
    # Note: We need the raw transaction df for recency calculation inside your old function
    # But our new aggregate function handles recency. 
    # Let's assume create_rfm_risk_label handles the logic to add 'is_high_risk'
    # For now, let's reuse your logic but ensure it fits the new flow.
    logger.info("Generating Proxy Target...")
    df_cust = create_rfm_risk_label(df_cust, df_raw) # Using your existing function logic
    
    # 3. Split
    drop_cols = ['is_high_risk', 'Cluster', 'FraudResult_max']
    X = df_cust.drop(columns=drop_cols) 
    y = df_cust['is_high_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Define Pipeline (Preprocessor + Model)
    # This is the "Production Grade" way: The pipeline contains the encoder!
    pipeline = Pipeline([
        ('preprocessor', CreditRiskPreprocessor()),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
    ])
    
    # 5. Train
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name="RF_Pipeline"):
        logger.info("Training Pipeline...")
        pipeline.fit(X_train, y_train)
        
        # Predict
        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:, 1]
        
        # Metrics
        roc_auc = roc_auc_score(y_test, probs)
        f1 = f1_score(y_test, preds)
        
        logger.info(f"Training Complete. AUC: {roc_auc:.4f}")
        
        # Log
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(pipeline, "model")
        
        # SAVE LOCAL ARTIFACTS (For API without MLflow dependency if needed)
        save_object(pipeline, os.path.join(ARTIFACT_PATH, "pipeline.pkl"))

        # Save the exact order of columns the model was trained on
        feature_names = X_train.columns.to_list()
        save_object(feature_names, os.path.join(ARTIFACT_PATH, "feature_names.pkl"))
        logger.info("✅ Feature names order saved.")

if __name__ == "__main__":
    main()