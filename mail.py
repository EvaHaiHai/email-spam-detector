# Assignment - Spam Detection Comparison (Full Workflow with Class Balance)

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

RANDOM_STATE = 42

# 1) Load dataset
df = pd.read_csv("dataset/mail_dataset.csv")
df = df.where(pd.notnull(df), "")
df.loc[df["Category"].str.lower().str.strip() == "spam", "Category"] = 0
df.loc[df["Category"].str.lower().str.strip() == "ham",  "Category"] = 1

X = df["Message"].astype(str)
y = df["Category"].astype(int)

# 2) Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# 3) TF-IDF vectorization
tfidf = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)
X_train_features = tfidf.fit_transform(X_train)
X_test_features  = tfidf.transform(X_test)

# 4) Define models with class_weight where possible
models = {
    "LogisticRegression": LogisticRegression(class_weight='balanced', random_state=RANDOM_STATE, max_iter=1000),
    "RandomForest": RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1),
    "NaiveBayes": MultinomialNB(),
    "SVM": LinearSVC(class_weight='balanced', random_state=RANDOM_STATE, max_iter=5000),
    "DecisionTree": DecisionTreeClassifier(class_weight='balanced', random_state=RANDOM_STATE)
}

# 5) Hyperparameter grids
param_grids = {
    "LogisticRegression": {"C": [0.01, 0.1, 1, 10], "solver": ["liblinear","lbfgs"], "max_iter": [1000,3000]},
    "RandomForest": {"n_estimators":[100,200,300], "max_depth":[None,10,20], "min_samples_split":[2,5,10]},
    "NaiveBayes": {"alpha":[0.01,0.1,1.0], "fit_prior":[True,False], "force_alpha":[True,False]},
    "SVM": {"C":[0.1,1,10], "loss":["hinge","squared_hinge"], "max_iter":[1000,3000]},
    "DecisionTree": {"max_depth":[None,10,20,30], "min_samples_split":[2,5,10], "criterion":["gini","entropy"]}
}

# 6) ---- TRAINING BEFORE TUNING (baseline) ----
baseline_preds = {}
print("\n=== TRAINING BASELINE MODELS (Before Tuning) ===")
for name, model in models.items():
    if name in ["RandomForest", "DecisionTree"]:
        model.fit(X_train_features.toarray(), y_train)
        y_pred = model.predict(X_test_features.toarray())
    else:
        model.fit(X_train_features, y_train)
        y_pred = model.predict(X_test_features)
    baseline_preds[name] = y_pred
    print(f"\n{name} (Baseline) Metrics:")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, pos_label=0))
    print("Recall   :", recall_score(y_test, y_pred, pos_label=0))
    print("F1-score :", f1_score(y_test, y_pred, pos_label=0))
    
# 7) ---- TRAINING AFTER TUNING ----
best_models = {}
tuned_preds = {}
print("\n=== GRID SEARCH AND TRAINING BEST MODELS (After Tuning) ===")
for name, model in models.items():
    print(f"\nTuning {name}...")
    grid = GridSearchCV(estimator=model, param_grid=param_grids[name], scoring="f1", cv=5, n_jobs=-1)
    if name in ["RandomForest", "DecisionTree"]:
        grid.fit(X_train_features.toarray(), y_train)
        y_pred = grid.predict(X_test_features.toarray())
    else:
        grid.fit(X_train_features, y_train)
        y_pred = grid.predict(X_test_features)
    best_models[name] = grid.best_estimator_
    tuned_preds[name] = y_pred
    print(f"Best {name} parameters: {grid.best_params_}")
    print("Test Set F1 Score:", f1_score(y_test, y_pred, pos_label=0))

# 8) ---- Performance Comparison ----
def print_confmat(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    cm_df = pd.DataFrame(cm, index=["Actual Ham","Actual Spam"], columns=["Pred Ham","Pred Spam"])
    print(f"\n{name} Confusion Matrix:\n{cm_df}")

print("\n=== PERFORMANCE COMPARISON (Tuned Models) ===")
for name, y_pred in tuned_preds.items():
    print(f"\n{name} Metrics:")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, pos_label=0))
    print("Recall   :", recall_score(y_test, y_pred, pos_label=0))
    print("F1-score :", f1_score(y_test, y_pred, pos_label=0))
    print_confmat(name, y_test, y_pred)

# 9) ---- Sample Messages Testing ----
samples = [
    "Congratulations! You've won a $1,000 Walmart gift card. Click here to claim your prize.",
    "Hey, are we still on for lunch tomorrow?",
    "URGENT! Your account has been compromised. Please reset your password immediately.",
    "Don't forget to bring the documents for our meeting next week.",
    "Buy prescription medications online at half the price. Limited stock available! Visit our site now.",
    "Subject: Make Money Fast – Secret Investment!, From: finance@quickcash.com, To: user@example.com, Invest $100 today and earn $1,000 in just one week. Don’t miss this opportunity! Click here to get started, Learn more: http://fake-investment-link.com"
]

def lab2str(v): return "Spam (0)" if v==0 else "Ham (1)"

print("\n=== SAMPLE MESSAGES PREDICTIONS (Tuned Models) ===")
for text in samples:
    print("\nText snippet:", text)
    for name, model in best_models.items():
        if name in ["RandomForest","DecisionTree"]:
            pred = model.predict(tfidf.transform([text]).toarray())[0]
        else:
            pred = model.predict(tfidf.transform([text]))[0]
        print(f"{name} Pred:", lab2str(pred))

# 10) ---- Save models and vectorizer ----
for name, model in best_models.items():
    joblib.dump(model, f"models/tuned_{name.lower()}_model.joblib")
joblib.dump(tfidf, "models/tfidf_vectorizer.joblib")
print("\nAll tuned models and vectorizer saved successfully.")
