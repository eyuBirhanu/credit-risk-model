import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from category_encoders import WOEEncoder
from typing import List

class CreditRiskPreprocessor(BaseEstimator, TransformerMixin):
    """
    A unified pipeline step that handles:
    1. Feature Aggregation (if raw data is passed)
    2. WoE Encoding
    3. Scaling (Optional, but good for some models)
    """
    
    def __init__(self, target_col='is_high_risk'):
        self.target_col = target_col
        self.woe_encoder = WOEEncoder(cols=['ProductCategory', 'ChannelId', 'PricingStrategy'], regularization=1.0)
        self.cat_cols = ['ProductCategory', 'ChannelId', 'PricingStrategy']
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Fits the WOE Encoder. 
        X must contain the categorical columns.
        y is required for WoE (supervised encoding).
        """
        # Ensure we are working with the customer-level aggregate data here
        # (Assuming X is already aggregated by CustomerId)
        self.woe_encoder.fit(X[self.cat_cols], y)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies transformations.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted yet!")
            
        X_out = X.copy()
        
        # Apply WoE
        X_encoded = self.woe_encoder.transform(X_out[self.cat_cols])
        
        # Rename columns to indicate transformation
        X_encoded = X_encoded.add_suffix('_WOE')
        
        # Drop original categoricals and join new ones
        X_out = X_out.drop(columns=self.cat_cols)
        X_out = pd.concat([X_out, X_encoded], axis=1)
        
        return X_out

def aggregate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Raw Transaction Data -> Customer Profile
    This function remains functional as it doesn't require 'fitting'.
    """
    # 1. Basic Aggregations
    aggs = {
        'Value': ['sum', 'mean', 'std', 'count'],
        'FraudResult': 'max'
    }
    
    cust_df = df.groupby('CustomerId').agg(aggs)
    cust_df.columns = ['_'.join(col).strip() for col in cust_df.columns.values]
    
    cust_df = cust_df.rename(columns={
        'Value_sum': 'Total_Spend',
        'Value_mean': 'Avg_Transaction_Value',
        'Value_count': 'Transaction_Frequency',
        'Value_std': 'Transaction_Variability'
    })
    
    # 2. Extract Most Frequent Categories (Mode)
    mode_func = lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
    cat_aggs = df.groupby('CustomerId').agg({
        'ProductCategory': mode_func,
        'ChannelId': mode_func,
        'PricingStrategy': mode_func
    })
    
    final_df = pd.concat([cust_df, cat_aggs], axis=1)
    
    # Fill NaN (Variability is NaN if only 1 transaction)
    final_df['Transaction_Variability'] = final_df['Transaction_Variability'].fillna(0)
    
    # Recency Calculation
    last_date = df['TransactionStartTime'].max()
    recency = df.groupby('CustomerId')['TransactionStartTime'].apply(lambda x: (last_date - x.max()).days)
    final_df['Recency'] = recency

    return final_df