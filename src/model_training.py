from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from data_preprocessing import load_data
import pickle

def train_model():
    X_train, X_test, y_train, y_test = load_data()
    
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    lr = LogisticRegression(max_iter=500)
    lr.fit(X_train_vec, y_train)
    lr_pred = lr.predict(X_test_vec)
    lr_acc = accuracy_score(y_test, lr_pred)
    print(f"Logistic Regression Accuracy: {lr_acc}")
    
    
    svm = LinearSVC()
    svm.fit(X_train_vec, y_train)
    svm_pred = svm.predict(X_test_vec)
    svm_acc = accuracy_score(y_test, svm_pred)
    print(f"SVM Accuracy: {svm_acc}")
    
    
    pickle.dump(vectorizer, open("artifacts/vectorizer.pkl", "wb"))

    # Save models
    pickle.dump(lr, open("artifacts/logistic_regression.pkl", "wb"))
    pickle.dump(svm, open("artifacts/svm_model.pkl", "wb"))

    print("\nModels saved successfully!")

    # Return best model
    if svm_acc > lr_acc:
        print("Best Model: SVM")
        best_model = svm
    else:
        print("Best Model: Logistic Regression")
        best_model = lr

    return best_model, X_test_vec, y_test


if __name__ == "__main__":
    train_model()

    