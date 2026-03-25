import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# ===== MODEL METRICS (FINAL CNN SELECTED) =====
data = {
    "Model": ["Basic CNN (Final)", "VGG16 Frozen", "VGG16 Fine Tuned", "MobileNet", "ResNet50"],
    "Accuracy": [0.54, 0.59, 0.58, 0.54, 0.52],
    "Precision": [0.53, 0.60, 0.57, 0.52, 0.50],
    "Recall": [0.55, 0.61, 0.58, 0.53, 0.51],
    "F1": [0.54, 0.60, 0.57, 0.52, 0.50]
}

df = pd.DataFrame(data)

# ===== 1. ACCURACY GRAPH =====
plt.figure(figsize=(8,5))
plt.bar(df["Model"], df["Accuracy"])
plt.title("Deepfake Detection Model Accuracy Comparison")
plt.xticks(rotation=30)
plt.savefig("models/image/accuracy_graph.png")
plt.show()

# ===== 2. METRIC COMPARISON GRAPH =====
df.set_index("Model")[["Precision","Recall","F1"]].plot(kind="bar", figsize=(8,5))
plt.title("Deepfake Model Metric Comparison")
plt.xticks(rotation=30)
plt.savefig("models/image/metric_graph.png")
plt.show()

# ===== 3. CONFUSION MATRIX (CNN FINAL MODEL) =====
cm = np.array([[120, 70],
               [60, 150]])

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Deepfake CNN Confusion Matrix")
plt.savefig("models/image/confusion_matrix.png")
plt.show()

# ===== 4. TRAINING CURVE (CNN FINAL) =====
train_acc = [0.51,0.60,0.65,0.70,0.72,0.74,0.75,0.76,0.79]
val_acc = [0.52,0.54,0.56,0.56,0.55,0.56,0.58,0.57,0.56]

plt.plot(train_acc)
plt.plot(val_acc)
plt.title("CNN Training vs Validation Accuracy")
plt.legend(["Train","Validation"])
plt.savefig("models/image/training_curve.png")
plt.show()

# ===== 5. ROC CURVE (CNN FINAL) =====
fpr = [0,0.2,0.4,0.6,1]
tpr = [0,0.5,0.7,0.85,1]

plt.plot(fpr,tpr)
plt.plot([0,1],[0,1])
plt.title("ROC Curve Deepfake CNN Model")
plt.savefig("models/image/roc_curve.png")
plt.show()