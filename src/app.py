from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import re

app = FastAPI(
    title="Fake News Detection API",
    description="Predict whether a given news text is Fake or Real",
    version="1.0"
)

# -----------------------
# Load Vectorizer + Model
# -----------------------
vectorizer = pickle.load(open("artifacts/vectorizer.pkl", "rb"))

# Try to load SVM first, fallback to logistic regression
try:
    model = pickle.load(open("artifacts/svm_model.pkl", "rb"))
    model_name = "SVM Model"
except:
    model = pickle.load(open("artifacts/logistic_regression.pkl", "rb"))
    model_name = "Logistic Regression Model"

print(f"Loaded Model: {model_name}")


# -----------------------
# Clean input text
# -----------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text


# -----------------------
# Request Body Schema
# -----------------------
class NewsItem(BaseModel):
    text: str


# -----------------------
# Routes
# -----------------------
@app.get("/")
def home():
    return {"message": "Fake News Detection API is Running!", "model": model_name}


@app.post("/predict")
def predict(item: NewsItem):
    cleaned = clean_text(item.text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    result = "Real News" if prediction == 1 else "Fake News"

    return {
        "input_text": item.text,
        "prediction": result,
        "model_used": model_name
    }
