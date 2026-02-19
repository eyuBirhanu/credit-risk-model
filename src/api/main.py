from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import logging
import os
from .pydantic_models import CreditScoringRequest, CreditScoringResponse

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Bati Bank Credit Scoring API")

# Global variable to hold the model
model = None

@app.on_event("startup")
def load_model():
    """
    On startup, load the pipeline directly from the local file.
    This is more robust than relying on MLflow's dynamic paths.
    """
    global model
    try:
        # Define path to the saved pipeline.pkl
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up two levels (from src/api -> src -> root) then into models/artifacts
        model_path = os.path.join(current_dir, '..', '..', 'models', 'artifacts', 'pipeline.pkl')
        model_path = os.path.abspath(model_path)
        
        logger.info(f"Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Did you run src/train.py?")

        # Load the model
        model = joblib.load(model_path)
        logger.info("✅ Model loaded successfully from local artifact.")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        # We don't raise e here to allow the API to start, 
        # but predictions will fail if model is None
        pass

@app.get("/")
def health_check():
    status = "active" if model is not None else "inactive (model missing)"
    return {"status": status}

@app.post("/predict", response_model=CreditScoringResponse)
def predict_credit_risk(request: CreditScoringRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")
    
    try:
        # 1. Convert request to DataFrame
        input_data = pd.DataFrame([request.model_dump()])
        
        # 2. Predict
        # The pipeline handles "Airtime" -> Numbers conversion automatically
        prob_high_risk = model.predict_proba(input_data)[0][1]
        
        # 3. Determine Label
        label = "High Risk" if prob_high_risk > 0.5 else "Low Risk"
        
        return {
            "risk_probability": round(prob_high_risk, 4),
            "risk_label": label,
            "model_version": "v2_production"
        }
        
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))