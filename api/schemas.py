from pydantic import BaseModel
from typing import Optional

class CreditRiskRequest(BaseModel):
    NAME_CONTRACT_TYPE: Optional[str]
    CODE_GENDER: Optional[str]
    FLAG_OWN_CAR: Optional[str]
    AMT_INCOME_TOTAL: Optional[float]
    AMT_CREDIT: Optional[float]
    NAME_INCOME_TYPE: Optional[str]
    NAME_EDUCATION_TYPE: Optional[str]
    OCCUPATION_TYPE: Optional[str]
    CNT_CHILDREN: Optional[int]
    DAYS_BIRTH: Optional[int]
    DAYS_EMPLOYED: Optional[int]
    EXT_SOURCE_1: Optional[float]
    EXT_SOURCE_2: Optional[float]
    EXT_SOURCE_3: Optional[float]
    REGION_RATING_CLIENT: Optional[int]

class CreditRiskResponse(BaseModel):
    risk_score: float
    risk_label: str
