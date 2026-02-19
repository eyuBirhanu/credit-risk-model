import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from category_encoders import WOEEncoder
import logging
import os

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(filepath: str) -> pd.DataFrame:
    """Loads raw data and converts timestamps."""
    logger.info(f"Loading data from {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    df = pd.read_csv(filepath)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    return df

def aggregate_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates transaction data into customer profiles."""
    logger.info("Aggregating transaction data into customer profiles...")
    
    # 1. Basic Aggregations
    aggs = {
        'Value': ['sum', 'mean', 'std', 'count'],
        'Amount': ['sum', 'mean'],
        'FraudResult': 'max'
    }
    
    cust_df = df.groupby('CustomerId').agg(aggs)
    cust_df.columns = ['_'.join(col).strip() for col in cust_df.columns.values]
    
    # Rename for clarity
    cust_df = cust_df.rename(columns={
        'Value_sum': 'Total_Spend',
        'Value_mean': 'Avg_Transaction_Value',
        'Value_count': 'Transaction_Frequency',
        'Value_std': 'Transaction_Variability'
    })
    
    # 2. Extract Most Frequent Categorical features
    mode_func = lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
    
    cat_aggs = df.groupby('CustomerId').agg({
        'ProductCategory': mode_func,
        'ChannelId': mode_func,
        'PricingStrategy': mode_func
    })
    
    final_df = pd.concat([cust_df, cat_aggs], axis=1)
    
    # Fill NaN values (e.g., std dev is NaN if only 1 transaction)
    final_df['Transaction_Variability'] = final_df['Transaction_Variability'].fillna(0)
    
    # Ensure Categorical Columns are strings
    final_df['ProductCategory'] = final_df['ProductCategory'].astype(str)
    final_df['ChannelId'] = final_df['ChannelId'].astype(str)
    final_df['PricingStrategy'] = final_df['PricingStrategy'].astype(str)
    
    return final_df

def create_rfm_risk_label(df: pd.DataFrame, transaction_df: pd.DataFrame) -> pd.DataFrame:
    """Creates the Proxy Target Variable 'is_high_risk' using RFM Analysis."""
    logger.info("Performing RFM Analysis and Clustering...")
    
    # Calculate Recency
    last_date = transaction_df['TransactionStartTime'].max()
    recency = transaction_df.groupby('CustomerId')['TransactionStartTime'].apply(lambda x: (last_date - x.max()).days)
    
    df['Recency'] = recency
    
    # Scale Data
    rfm_cols = ['Recency', 'Transaction_Frequency', 'Total_Spend']
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(df[rfm_cols])
    
    # Clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Identify High Risk Cluster (Lowest Frequency)
    centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=rfm_cols)
    centers['Cluster'] = [0, 1, 2]
    bad_cluster_idx = centers.sort_values(by='Transaction_Frequency').iloc[0]['Cluster']
    
    logger.info(f"Identified Cluster {bad_cluster_idx} as High Risk (Proxy Default).")
    
    df['is_high_risk'] = df['Cluster'].apply(lambda x: 1 if x == bad_cluster_idx else 0)
    
    return df

def calculate_iv(df, feature, target):
    """Auxiliary function to calculate Information Value (IV)."""
    lst = []
    for i in range(df[feature].nunique()):
        val = list(df[feature].unique())[i]
        lst.append([feature, val, df[df[feature] == val].count()[feature], df[(df[feature] == val) & (df[target] == 0)].count()[feature], df[(df[feature] == val) & (df[target] == 1)].count()[feature]])
    
    data = pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Good', 'Bad'])
    data['Share'] = data['All'] / data['All'].sum()
    data['Bad Rate'] = data['Bad'] / data['All']
    data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())
    data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()
    data['WoE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])
    
    data = data.replace({'WoE': {np.inf: 0, -np.inf: 0}})
    data['IV'] = data['WoE'] * (data['Distribution Good'] - data['Distribution Bad'])
    
    iv = data['IV'].sum()
    return iv

def apply_woe_iv(df: pd.DataFrame, target: str = 'is_high_risk'):
    """Applies WoE using category_encoders and calculates IV manually."""
    logger.info("Calculating WoE and IV for features...")
    
    cat_cols = ['ProductCategory', 'ChannelId', 'PricingStrategy']
    
    # 1. Calculate and Print IV (for Report)
    logger.info("--- Information Value (IV) Stats ---")
    for col in cat_cols:
        try:
            iv_val = calculate_iv(df, col, target)
            logger.info(f"Feature: {col} | IV: {iv_val:.4f}")
        except Exception as e:
            logger.warning(f"Could not calculate IV for {col}: {e}")

    # 2. Apply WoE Transformation (using Robust Encoder)
    # regularization=1.0 prevents division by zero
    enc = WOEEncoder(cols=cat_cols, regularization=1.0)
    
    # Create new WoE columns
    df_woe = enc.fit_transform(df[cat_cols], df[target])
    df_woe = df_woe.add_suffix('_WOE')
    
    # Join back
    df = pd.concat([df, df_woe], axis=1)
    
    return df

def process_pipeline(raw_filepath: str, output_filepath: str):
    """Main function to run the pipeline."""
    # Load
    raw_df = load_data(raw_filepath)
    
    # Aggregate
    cust_df = aggregate_customer_features(raw_df)
    
    # Create Target
    cust_df = create_rfm_risk_label(cust_df, raw_df)
    
    # Feature Engineering
    cust_df = apply_woe_iv(cust_df)
    
    # Save
    cust_df.to_csv(output_filepath, index=True)
    logger.info(f"Processed data saved to {output_filepath}")

if __name__ == "__main__":
    # Robust path handling
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(current_dir, '..', 'data', 'raw', 'data.csv')
    processed_data_path = os.path.join(current_dir, '..', 'data', 'processed', 'training.csv')
    
    # Ensure processed directory exists
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    
    try:
        process_pipeline(raw_data_path, processed_data_path)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")