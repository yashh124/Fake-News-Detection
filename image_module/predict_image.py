import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("models/image/cnn_model.keras")

# Image size used during training
IMG_SIZE = (128, 128)

def predict_image(img_path):

    # Load image
    img = image.load_img(img_path, target_size=IMG_SIZE)

    # Convert to array
    img_array = image.img_to_array(img)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)

    probability = prediction[0][0]

    if probability > 0.5:
        return "Fake Image", probability
    else:
        return "Real Image", probability


# Testing directly
if __name__ == "__main__":

    test_image = "test.jpg"   # put any test image here

    label, prob = predict_image(test_image)

    print("Prediction:", label)
    print("Confidence:", prob)