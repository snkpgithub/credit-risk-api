from fastapi import FastAPI
from api.schemas import CreditRiskRequest, CreditRiskResponse
from model.predict import predict_credit_risk
from explain.shap_explainer import get_shap_explanation
from typing import List

app = FastAPI(
    title="Real-Time Credit Risk API",
    version="1.0"
)

@app.post("/predict", response_model=CreditRiskResponse)
def predict(request: CreditRiskRequest):
    """
    Predict credit risk score and label.
    """
    input_dict = request.dict(exclude_none=True)
    result = predict_credit_risk(input_dict)
    return CreditRiskResponse(
        risk_score=float(result["risk_score"]),
        risk_label=result["risk_label"]
    )

@app.post("/explain")
def explain(request: CreditRiskRequest) -> List[dict]:
    """
    Generate SHAP-based explanation of model decision.
    """
    input_dict = request.dict(exclude_none=True)
    explanation = get_shap_explanation(input_dict)
    return explanation
