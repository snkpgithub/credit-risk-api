#  Real-Time Credit Risk API

A FastAPI-based backend service for predicting credit default risk using XGBoost. This system also provides SHAP-based feature attributions to explain model decisions.

---

##  Features

* Predicts credit default risk score from applicant data
* Classifies risk as "High Risk" or "Low Risk"
* Returns top SHAP features contributing to decision
* Clean API with Swagger documentation

---

## 📅 Tech Stack

* **FastAPI** - Web framework
* **XGBoost** - Model backend
* **SHAP** - Model explanation
* **Scikit-learn** - Preprocessing
* **Uvicorn** - ASGI server

---

## 🔧 Installation

```bash
# Clone repo
git clone https://github.com/snkpgithub/credit-risk-api.git
cd credit-risk-api

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn api.main:app --reload
```

Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🤝 Endpoints

### `POST /predict`

Returns risk score and label.

```json
{
  "risk_score": 0.4573,
  "risk_label": "Low Risk"
}
```

### `POST /explain`

Returns top 5 SHAP feature attributions.

```json
[
  {"feature": "DAYS_ID_PUBLISH", "value": 1.98, "impact": 1.25},
  ...
]
```

---

## 💡 Example Request

```json
{
  "NAME_CONTRACT_TYPE": "Cash loans",
  "CODE_GENDER": "M",
  "FLAG_OWN_CAR": "N",
  "AMT_INCOME_TOTAL": 202500.0,
  "AMT_CREDIT": 406597.5,
  "NAME_INCOME_TYPE": "Working",
  "NAME_EDUCATION_TYPE": "Higher education",
  "OCCUPATION_TYPE": "Laborers",
  "CNT_CHILDREN": 0,
  "DAYS_BIRTH": -12005,
  "DAYS_EMPLOYED": -4542,
  "EXT_SOURCE_1": 0.1,
  "EXT_SOURCE_2": 0.4,
  "EXT_SOURCE_3": 0.5,
  "REGION_RATING_CLIENT": 2
}
```

---

## 📄 File Structure

```
credit-risk-api/
│
├── api/
│   └── main.py             # FastAPI app entry point
│
├── model/
│   └── predict.py          # Model prediction logic
│
├── explain/
│   └── shap_explainer.py   # SHAP-based explanation logic
│
├── models/                 # Model + preprocessing artifacts
├── requirements.txt
└── README.md
```

---

## 🌟 Author

**Shashank Pandey**
[GitHub](https://github.com/snkpgithub) | [LinkedIn](https://linkedin.com/in/snkp0018)

---

## 📉 Future Work

* Add frontend UI for input forms
* Deploy on AWS Lambda / EC2
* Add logging and monitoring (e.g. Prometheus, Grafana)
* Add CI/CD pipeline

---

> Built with ❤️ for interpretable, real-time ML decisions in financial services
