import pytest
import pandas as pd
import numpy as np
from src.data_processing import aggregate_customer_features

def test_aggregate_customer_features():
    data = {
        'CustomerId': ['C1', 'C1', 'C2'],
        'TransactionId': ['T1', 'T2', 'T3'],
        'Value': [100, 200, 50],
        'Amount': [100, 200, 50],
        'FraudResult': [0, 0, 1],
        'ProductCategory': ['Airtime', 'Airtime', 'Data'],
        'ChannelId': ['Web', 'Web', 'App'],
        'PricingStrategy': [1, 1, 2],
        'TransactionStartTime': pd.to_datetime(['2025-01-01', '2025-01-02', '2025-01-03'])
    }
    df = pd.DataFrame(data)
    
    result = aggregate_customer_features(df)
    
    assert result.shape[0] == 2 
    assert result.loc['C1', 'Total_Spend'] == 300 
    assert result.loc['C2', 'FraudResult_max'] == 1 
    assert 'Transaction_Variability' in result.columns