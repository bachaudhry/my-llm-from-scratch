# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# from sklearn.metrics import balanced_accuracy_score
from sklearn.dummy import DummyClassifier


def load_dataframes():
    df_train = pd.read_csv("data/train.csv")
    df_val = pd.read_csv("data/validation.csv")
    df_test = pd.read_csv("data/test.csv")

    return df_train, df_val, df_test


def eval_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    # Making predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    # Calculating accuracy and balanced accuracy
    acc_train = accuracy_score(y_train, y_pred_train)
    # balanced_accuracy_train = balanced_accuracy_score(y_train, y_pred_train)

    acc_val = accuracy_score(y_val, y_pred_val)
    # balanced_accuracy_val = balanced_accuracy_score(y_val, y_pred_val)

    acc_test = accuracy_score(y_test, y_pred_test)
    # balanced_accuracy_test = balanced_accuracy_score(y_test, y_pred_test)

    # Printing the results
    print(f"Training Accuracy: {acc_train*100:.2f}%")
    print(f"Validation Accuracy: {acc_val*100:.2f}%")
    print(f"Test Accuracy: {acc_test*100:.2f}%")

    # print(f"\nTraining Balanced Accuracy: {balanced_accuracy_train*100:.2f}%")
    # print(f"Validation Balanced Accuracy: {balanced_accuracy_val*100:.2f}%")
    # print(f"Test Balanced Accuracy: {balanced_accuracy_test*100:.2f}%")


if __name__ == "__main__":
    df_train, df_val, df_test = load_dataframes()

    #########################################
    # Convert text into bag-of-words model
    vectorizer = CountVectorizer()
    #########################################

    X_train = vectorizer.fit_transform(df_train["text"])
    X_val = vectorizer.transform(df_val["text"])
    X_test = vectorizer.transform(df_test["text"])
    y_train, y_val, y_test = df_train["label"], df_val["label"], df_test["label"]

    #####################################
    # Model training and evaluation
    #####################################

    # Create a dummy classifier with the strategy to predict the most frequent class
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, y_train)

    print("Dummy classifier:")
    eval_model(dummy_clf, X_train, y_train, X_val, y_val, X_test, y_test)

    print("\n\nLogistic regression classifier:")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    eval_model(model, X_train, y_train, X_val, y_val, X_test, y_test)