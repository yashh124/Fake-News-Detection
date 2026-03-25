import pandas as pd
import os
import matplotlib.pyplot as plt

# ================= TEXT DATASET PIE =================

true = pd.read_csv("data/text/True.csv")
fake = pd.read_csv("data/text/Fake.csv")

text_counts = [len(true), len(fake)]
labels = ["Real News", "Fake News"]

plt.figure(figsize=(5,5))
plt.pie(text_counts, labels=labels, autopct='%1.1f%%')
plt.title("Text Dataset Distribution")
plt.savefig("models/text/text_dataset_pie.png")
plt.show()

# ================= IMAGE DATASET PIE =================

image_path = "data/image"

fake_images = len(os.listdir(os.path.join(image_path, "fake")))
real_images = len(os.listdir(os.path.join(image_path, "real")))

image_counts = [real_images, fake_images]
labels = ["Real Images", "Fake Images"]

plt.figure(figsize=(5,5))
plt.pie(image_counts, labels=labels, autopct='%1.1f%%')
plt.title("Image Dataset Distribution")
plt.savefig("models/image/image_dataset_pie.png")
plt.show()

print("Text Dataset:", text_counts)
print("Image Dataset:", image_counts)