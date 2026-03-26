import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

print("Loading dataset...")

# Load datasets
df_true = pd.read_csv("data/text/True.csv")
df_fake = pd.read_csv("data/text/Fake.csv")
df_true["label"] = 1
df_fake["label"] = 0

df = pd.concat([df_true, df_fake], axis=0)

# Combine title + text (IMPORTANT)
df["content"] = df["title"] + " " + df["text"]

df = df[["content", "label"]]
df = df.sample(frac=1, random_state=42)

print("Label distribution:")
print(df["label"].value_counts())

# Train test split
X = df["content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    max_features=50000   # GOOD BALANCE
)

print("Vectorizing...")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train SVM
print("Training SVM...")
svm = LinearSVC()
svm.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = svm.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(svm, "svm_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model saved successfully!")