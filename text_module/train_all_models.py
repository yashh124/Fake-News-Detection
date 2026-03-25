import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

# LOAD DATA
true = pd.read_csv("data/text/True.csv")
fake = pd.read_csv("data/text/Fake.csv")

true["label"] = 1
fake["label"] = 0

df = pd.concat([true, fake])
df = df.sample(frac=1)

X = df["text"]
y = df["label"]

# TFIDF
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X = tfidf.fit_transform(X)

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# MODELS
models = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "Passive Aggressive": PassiveAggressiveClassifier(),
    "Extra Trees": ExtraTreesClassifier()
}

results = []

for name, model in models.items():
    print("Training:", name)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    pre = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    results.append([name, acc, pre, rec, f1])

results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])

print(results_df)

# SAVE RESULTS
results_df.to_csv("models/text/text_model_results.csv", index=False)

# GRAPH
results_df.set_index("Model").plot(kind="bar", figsize=(12,6))
plt.title("Fake News Model Comparison")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("models/text/text_comparison_graph.png")
plt.show()