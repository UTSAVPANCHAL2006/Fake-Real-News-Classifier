import pandas as pd 
import re
import nltk
from sklearn.model_selection import train_test_split

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

def load_data(fake_path="/Users/utsav/Desktop/dlpro/News _dataset/data/Fake.csv",
                true_path="/Users/utsav/Desktop/dlpro/News _dataset/data/True.csv",):
    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)
    
    fake["label"] = 0
    true["label"] = 1 
    
    df = pd.concat([fake, true], axis=0).reset_index(drop=True)
    df["final_text"] = (df["title"] + " " + df["text"]).apply(clean_text)

    X = df["final_text"]
    y = df["label"]

    return train_test_split(X, y, test_size=0.2, random_state=42)
    
    
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    print("Data loaded successfully.")

    