import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Load datasets
df_true = pd.read_csv("True.csv")
df_fake = pd.read_csv("Fake.csv")

# Add labels
df_true["label"] = 1
df_fake["label"] = 0

# Combine datasets
df = pd.concat([df_true, df_fake], axis=0)

# Combine title + text
df["content"] = df["title"] + " " + df["text"]

# Keep only required columns
df = df[["content", "label"]]

# Shuffle dataset
df = df.sample(frac=1, random_state=42)

print("Dataset shape:", df.shape)

# Split dataset
X = df["content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training size:", X_train.shape)
print("Testing size:", X_test.shape)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF shape:", X_train_tfidf.shape)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Train Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_tfidf, y_train)

# Predictions
y_pred_lr = lr.predict(X_test_tfidf)

# Evaluation
accuracy_lr = accuracy_score(y_test, y_pred_lr)

print("Logistic Regression Accuracy:", accuracy_lr)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_lr))


from sklearn.naive_bayes import MultinomialNB

# Train Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)

# Predictions
y_pred_nb = nb.predict(X_test_tfidf)

# Evaluation
accuracy_nb = accuracy_score(y_test, y_pred_nb)

print("\nNaive Bayes Accuracy:", accuracy_nb)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_nb))


from sklearn.svm import LinearSVC

# Train SVM
svm = LinearSVC()
svm.fit(X_train_tfidf, y_train)

# Predictions
y_pred_svm = svm.predict(X_test_tfidf)

# Evaluation
accuracy_svm = accuracy_score(y_test, y_pred_svm)

print("\nSVM Accuracy:", accuracy_svm)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_svm))

# from sklearn.model_selection import cross_val_score

# # 5-Fold Cross Validation for SVM
# cv_scores = cross_val_score(svm, X_train_tfidf, y_train, cv=5)

# print("\nCross Validation Scores:", cv_scores)
# print("Average CV Accuracy:", cv_scores.mean())


# from sklearn.ensemble import RandomForestClassifier

# # Train Random Forest
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(X_train_tfidf, y_train)

# # Predictions
# y_pred_rf = rf.predict(X_test_tfidf)

# # Evaluation
# accuracy_rf = accuracy_score(y_test, y_pred_rf)

# print("\nRandom Forest Accuracy:", accuracy_rf)
# print("\nClassification Report:\n")
# print(classification_report(y_test, y_pred_rf))


# from sklearn.metrics import roc_curve, auc
# import matplotlib.pyplot as plt

# # Get decision scores (not predict)
# y_scores = svm.decision_function(X_test_tfidf)

# # Compute ROC curve
# fpr, tpr, thresholds = roc_curve(y_test, y_scores)
# roc_auc = auc(fpr, tpr)

# # Plot ROC Curve
# plt.figure()
# plt.plot(fpr, tpr, label="SVM (AUC = %0.2f)" % roc_auc)
# plt.plot([0, 1], [0, 1], linestyle='--')
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve - SVM")
# plt.legend()
# plt.show()



import joblib

joblib.dump(svm, "svm_model.pkl", compress=3)
joblib.dump(vectorizer, "tfidf_vectorizer.pkl", compress=3)

# # Save SVM model
# joblib.dump(svm, "svm_model.pkl")

# # Save TF-IDF vectorizer
# joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# print("Model and Vectorizer saved successfully!")