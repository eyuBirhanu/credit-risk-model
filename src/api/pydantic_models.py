from pydantic import BaseModel, Field

class CreditScoringRequest(BaseModel):
    # --- NUMERICAL FEATURES ---
    # These are the raw numbers, same as before
    Total_Spend: float = Field(..., description="Sum of all transaction values")
    Avg_Transaction_Value: float = Field(..., description="Average value per transaction")
    Transaction_Frequency: float = Field(..., description="Total number of transactions")
    Transaction_Variability: float = Field(..., description="Standard deviation of transaction amounts")
    Recency: float = Field(..., description="Days since last transaction")
    
    # --- CATEGORICAL FEATURES (THE UPDATE) ---
    # OLD WAY: We asked for float (e.g., ProductCategory_WOE)
    # NEW WAY: We ask for string (e.g., "Airtime", "Data", "Financial Services")
    ProductCategory: str = Field(..., example="Airtime")
    ChannelId: str = Field(..., example="Android")
    PricingStrategy: str = Field(..., example="Category_1")

class CreditScoringResponse(BaseModel):
    risk_probability: float
    risk_label: str
    model_version: str