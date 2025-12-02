from model_training import train_model
from sklearn.metrics import classification_report, confusion_matrix , accuracy_score


def evaluate_model():
    best_model, X_test_vec, y_test = train_model()
    
    y_pred = best_model.predict(X_test_vec)
    
    acc = accuracy_score(y_test, y_pred)
    print("\n============================")
    print(f"Accuracy: {acc}")
    print("============================\n")
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
if __name__ == "__main__":
    evaluate_model()