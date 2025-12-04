import pickle
import re

vectorizer = pickle.load(open("artifacts/vectorizer.pkl", "rb"))
try:
    model = pickle.load(open("artifacts/svm_model.pkl", "rb"))
    best_model_name = "SVM"
except:
    model = pickle.load(open("artifacts/logistic_regression.pkl", "rb"))
    best_model_name = "Logistic Regression"

print(f"Loaded Best Model: {best_model_name}")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

def predict_news(text):
    clean = clean_text(text)
    vector = vectorizer.transform([clean])
    pred = model.predict(vector)[0]

    return "Real News" if pred == 1 else "Fake News"


if __name__ == "__main__":
    sample = input("Enter news text: ")
    print("Prediction:", predict_news(sample))