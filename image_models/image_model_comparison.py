import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, VGG16, ResNet50
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

IMG_SIZE = 128
BATCH = 16

train_dir = "data/image"

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

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
    subset='validation'
)

results = []

def train_and_eval(model, name):
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(train, validation_data=val, epochs=5, verbose=1)
    acc = max(history.history['val_accuracy'])
    results.append([name, acc])

# BASIC CNN
cnn = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(IMG_SIZE,IMG_SIZE,3)),
    MaxPooling2D(),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128,activation='relu'),
    Dense(1,activation='sigmoid')
])
train_and_eval(cnn,"Basic CNN")

# MOBILENET
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
base.trainable=False
x = GlobalAveragePooling2D()(base.output)
x = Dense(1,activation='sigmoid')(x)
mobilenet = Model(base.input,x)
train_and_eval(mobilenet,"MobileNet")

# VGG
base = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
base.trainable=False
x = GlobalAveragePooling2D()(base.output)
x = Dense(1,activation='sigmoid')(x)
vgg = Model(base.input,x)
train_and_eval(vgg,"VGG16")

# RESNET
base = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
base.trainable=False
x = GlobalAveragePooling2D()(base.output)
x = Dense(1,activation='sigmoid')(x)
resnet = Model(base.input,x)
train_and_eval(resnet,"ResNet50")

df = pd.DataFrame(results, columns=["Model","Val Accuracy"])
df.to_csv("models/image/image_results.csv",index=False)

df.plot(x="Model",y="Val Accuracy",kind="bar")
plt.title("Deepfake Model Comparison")
plt.show()