import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Model": [
        "Basic CNN",
        "MobileNet",
        "VGG16 (Frozen)",
        "ResNet50",
        "VGG16 (Fine Tuned)"
    ],
    "Accuracy": [
        0.54,
        0.54,
        0.59,
        0.52,
        0.58
    ]
}

df = pd.DataFrame(data)
df = df.sort_values(by="Accuracy", ascending=False)

# SAVE CSV
df.to_csv("models/image/final_image_results.csv", index=False)

# TABLE FIGURE
fig, ax = plt.subplots(figsize=(8,4))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2,1.2)

plt.title("Deepfake Detection Model Comparison")
plt.savefig("models/image/final_image_table.png")
plt.show()