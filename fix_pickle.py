import joblib

print("Loading old models...")

model = joblib.load("models/text/svm_model.pkl")
vectorizer = joblib.load("models/text/tfidf_vectorizer.pkl")

print("Resaving models with compatible format...")

joblib.dump(model, "models/text/svm_model.pkl")
joblib.dump(vectorizer, "models/text/tfidf_vectorizer.pkl")

print("DONE ✅")