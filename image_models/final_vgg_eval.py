import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

IMG_SIZE = 128
BATCH = 16

train_dir = "data/image"

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val = datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# LOAD TRAINED MODEL (SAVE FIRST FROM PREVIOUS CODE)
model = load_model("models/image/vgg_finetuned.h5")

pred = model.predict(val)
pred_classes = (pred > 0.5).astype(int)

cm = confusion_matrix(val.classes, pred_classes)
print(cm)

print(classification_report(val.classes, pred_classes))

plt.imshow(cm)
plt.title("Fine Tuned VGG Confusion Matrix")
plt.colorbar()
plt.show()

fpr, tpr, _ = roc_curve(val.classes, pred)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr)
plt.title("ROC Curve Deepfake")
plt.show()