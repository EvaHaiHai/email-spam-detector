# Discussing and Reflecting on Achieved Results of SMS Spam Detector

## Overview of the Project Process

This project focused on building an SMS Spam Detection system using machine learning techniques. The main objective was to train, evaluate, and compare multiple classifiers in order to identify a reliable model capable of accurately distinguishing spam messages from legitimate (ham) messages. The project followed a complete machine learning pipeline, including data preprocessing, feature engineering, baseline model training, hyperparameter tuning, evaluation, and reflection.

The dataset used consisted of approximately 5,500 SMS messages and exhibited a clear class imbalance, with ham messages significantly outnumbering spam messages. This imbalance represented a major challenge, as it could negatively affect model performance if not properly handled.

## Interpretation of Results

### Baseline Models Performance

Before hyperparameter tuning, all models were trained using baseline configurations. The results showed that Support Vector Machine (SVM) achieved the strongest overall baseline performance, with high accuracy and a balanced F1-score. Logistic Regression and Random Forest also performed well, although Random Forest showed high precision with comparatively lower recall, indicating that some spam messages were missed.

Naive Bayes demonstrated very high precision but suffered from lower recall, suggesting that while it was confident when predicting spam, it failed to detect a significant portion of spam messages. Decision Tree performed the weakest among the baseline models, highlighting its sensitivity to high-dimensional TF-IDF features.

These results indicated that baseline training alone was insufficient, especially in the presence of class imbalance.

### Impact of Hyperparameter Tuning

After applying GridSearchCV for hyperparameter tuning, the performance of most models improved noticeably. The tuned SVM emerged as the best-performing model overall, achieving a strong balance between precision and recall and the highest F1-score among all models. This balance is particularly important for spam detection, where missing spam messages can be costly.

Tuned Naive Bayes also showed significant improvement, demonstrating that even relatively simple models can become competitive when properly tuned. Logistic Regression and Random Forest benefited from tuning as well, while Decision Tree performance declined slightly, suggesting overfitting issues when applied to sparse text features.

### Confusion Matrix and Sample Predictions

Confusion matrix analysis confirmed that the best-performing models were effective at minimizing both false positives and false negatives. Sample message testing further validated the practical reliability of the models, as they correctly classified both spam-like and legitimate messages in realistic scenarios.

## Challenges Faced

One of the main challenges encountered was the class imbalance in the dataset, which required the use of class weight adjustments to prevent bias toward the majority class. Another challenge was the computational cost of hyperparameter tuning, especially for models such as Random Forest and SVM. Additionally, model selection required careful consideration of multiple evaluation metrics rather than relying solely on accuracy.

## Insights Gained

This project highlighted the effectiveness of TF-IDF for text classification tasks and demonstrated that SVM is particularly well-suited for medium-sized text datasets. It also reinforced the importance of hyperparameter tuning and the limitations of accuracy as a standalone evaluation metric when dealing with imbalanced data.
