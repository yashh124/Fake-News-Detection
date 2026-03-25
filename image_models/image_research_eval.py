import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam

IMG_SIZE = 128
BATCH = 16

train_dir = "data/image"

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train = datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode='binary',
    subset='training'
)

val = datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# BEST MODEL = VGG (based on your graph)
base = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
base.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = Dense(1, activation='sigmoid')(x)

model = Model(base.input, x)

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train, validation_data=val, epochs=5)

# PREDICTIONS
pred = model.predict(val)
pred = (pred > 0.5).astype(int)

cm = confusion_matrix(val.classes, pred)

print("CONFUSION MATRIX")
print(cm)

print("CLASSIFICATION REPORT")
print(classification_report(val.classes, pred))

plt.imshow(cm)
plt.title("Deepfake Confusion Matrix")
plt.colorbar()
plt.show()

# TRAINING CURVE
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Training vs Validation Accuracy")
plt.show()