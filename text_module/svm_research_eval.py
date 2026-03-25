import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_curve, auc

true = pd.read_csv("data/text/True.csv")
fake = pd.read_csv("data/text/Fake.csv")

true["label"] = 1
fake["label"] = 0

df = pd.concat([true, fake])
df = df.sample(frac=1, random_state=42)

X = df["text"]
y = df["label"]

tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LinearSVC()
model.fit(X_train, y_train)

# CROSS VALIDATION
cv_scores = cross_val_score(model, X, y, cv=5)

plt.plot(cv_scores)
plt.title("Cross Validation Accuracy (5 Fold)")
plt.ylabel("Accuracy")
plt.show()

# ROC
y_score = model.decision_function(X_test)
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr)
plt.title("ROC Curve SVM")
plt.show()