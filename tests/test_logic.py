import pandas as pd
import pytest
from src.features import aggregate_transactions, CreditRiskPreprocessor

# --- FIXTURES (Sample Data) ---
@pytest.fixture
def sample_raw_data():
    """Creates a small dataframe mimicking the raw transaction log."""
    data = {
        'TransactionId': ['T1', 'T2', 'T3', 'T4'],
        'BatchId': ['B1', 'B1', 'B2', 'B2'],
        'AccountId': ['A1', 'A1', 'A2', 'A2'],
        'SubscriptionId': ['S1', 'S1', 'S2', 'S2'],
        'CustomerId': ['C1', 'C1', 'C2', 'C2'], # Two customers
        'CurrencyCode': ['ETB', 'ETB', 'ETB', 'ETB'],
        'CountryCode': [251, 251, 251, 251],
        'ProviderId': ['P1', 'P1', 'P2', 'P2'],
        'ProductId': ['Prod1', 'Prod1', 'Prod2', 'Prod2'],
        'ProductCategory': ['Airtime', 'Airtime', 'Financial Services', 'Financial Services'],
        'ChannelId': ['Android', 'Android', 'Web', 'Web'],
        'Amount': [100.0, 50.0, 500.0, 1000.0],
        'Value': [100, 50, 500, 1000],
        'TransactionStartTime': [
            '2025-02-01T00:00:00Z', 
            '2025-02-01T01:00:00Z',
            '2025-02-10T12:00:00Z',
            '2025-02-10T13:00:00Z'
        ],
        'PricingStrategy': ['1', '1', '2', '2'],
        'FraudResult': [0, 0, 0, 0]
    }
    df = pd.DataFrame(data)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    return df

# --- TESTS ---

def test_aggregation_logic(sample_raw_data):
    """
    Risk Check: Does the aggregation correctly sum up money?
    If this fails, we might approve a user who actually has $0 spend.
    """
    df_agg = aggregate_transactions(sample_raw_data)
    
    # Check shape (2 customers)
    assert len(df_agg) == 2
    
    # Check Customer C1 logic
    # Spent 100 + 50 = 150
    assert df_agg.loc['C1', 'Total_Spend'] == 150.0
    assert df_agg.loc['C1', 'Transaction_Frequency'] == 2
    
    # Check Customer C2 logic
    # Spent 500 + 1000 = 1500
    assert df_agg.loc['C2', 'Total_Spend'] == 1500.0

def test_pipeline_handling_unknown_categories():
    """
    Risk Check: What happens if a user buys a product we have NEVER seen before?
    The model should NOT crash (Robustness).
    """
    # 1. Train a preprocessor on simple data
    train_df = pd.DataFrame({
        'ProductCategory': ['Airtime', 'Data', 'Airtime'],
        'ChannelId': ['App', 'App', 'Web'],
        'PricingStrategy': ['1', '1', '2']
    })
    y = pd.Series([0, 1, 0]) # Dummy target
    
    preprocessor = CreditRiskPreprocessor()
    preprocessor.fit(train_df, y)
    
    # 2. Test with a BRAND NEW category "CryptoCurrency"
    test_df = pd.DataFrame({
        'ProductCategory': ['CryptoCurrency'], # <--- NEW!
        'ChannelId': ['App'],
        'PricingStrategy': ['1']
    })
    
    try:
        result = preprocessor.transform(test_df)
        # If we reach here, it didn't crash.
        # Check if it assigned a neutral value (usually 0 or similar for WoE)
        assert 'ProductCategory_WOE' in result.columns
        print("\nSuccess: Pipeline handled unknown category without crashing.")
    except Exception as e:
        pytest.fail(f"Pipeline crashed on unknown category: {e}")