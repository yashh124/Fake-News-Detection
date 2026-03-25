import joblib

# -----------------------------
# Load Saved Model & Vectorizer
# -----------------------------
print("Loading model...")

model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

print("Model loaded successfully!")

# -----------------------------
# Prediction Function
# -----------------------------
def predict_news(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]

    if prediction == 1:
        return "Real News"
    else:
        return "Fake News"

# -----------------------------
# Run Prediction
# -----------------------------
if __name__ == "__main__":
    print("\nEnter news text (type 'exit' to quit):")

    while True:
        user_input = input("\nNews: ")

        if user_input.lower() == "exit":
            print("Exiting...")
            break

        result = predict_news(user_input)
        print("Prediction:", result)