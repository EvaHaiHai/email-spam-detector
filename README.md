# Email Spam Detector

Lightweight Flask UI/API for classifying email or SMS text as spam or ham. Models are trained and tuned with the provided script, stored as joblib artifacts, and served through a single-page UI or JSON endpoint.

## Contents
- app.py: Flask app (routes `/` and `/predict`), CORS enabled.
- templates/index.html: Tailwind-based UI for testing messages and choosing model versions.
- models/: Folder holding the tuned model artifacts and TF-IDF vectorizer used at runtime.
- mail.py: Full training and tuning pipeline (TF-IDF, multiple classifiers, GridSearchCV, metrics, artifact export).
- mail_dataset.csv: Labeled dataset (spam=0, ham=1).
- reflection_paper.md: Narrative on experiments and results.

## Quick Setup and Run
```bash
pip install -r requirements.txt

# Place artifacts where app.py expects them
mkdir -p models
cp models/tfidf_vectorizer.joblib models/
cp models/tuned_svm_model.joblib models/
cp models/tuned_logisticregression_model.joblib models/
cp models/tuned_naivebayes_model.joblib models/

# Start the server
python ./app.py
```
- UI: open http://127.0.0.1:7000/
- API example:
```bash
curl -X POST http://127.0.0.1:7000/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "Win a free prize now!", "version": "version-1"}'
```

### Model Versions (app.py)
- version-1 → tuned_svm_model.joblib
- version-2 → tuned_logisticregression_model.joblib
- version-3 → tuned_naivebayes_model.joblib

## Training Workflow (mail.py)
- Load data, clean text, label encode (spam=0, ham=1).
- Split stratified train/test; build TF-IDF features.
- Train baselines: Logistic Regression, Random Forest, Naive Bayes, SVM, Decision Tree.
- Hyperparameter tuning with GridSearchCV; pick best estimators.
- Report Accuracy, Precision, Recall, F1, confusion matrices; sample predictions.
- Save tuned models and vectorizer to models/ (joblib).

After retraining, ensure the latest artifacts remain in models/ before serving.

## Data
- Source: mail_dataset.csv (≈5.5k rows, spam/ham).
- Class imbalance addressed with `class_weight='balanced'` where supported.

## Results Snapshot
- From reflection_paper.md: tuned SVM delivered the best precision/recall balance and F1; tuned Naive Bayes improved notably after tuning.

## Troubleshooting
- TemplateNotFound: ensure templates/index.html exists (copy from templates/templates/ if missing).
- Missing artifacts: verify models/ contains tfidf_vectorizer.joblib, tuned_svm_model.joblib, tuned_logisticregression_model.joblib, tuned_naivebayes_model.joblib.
