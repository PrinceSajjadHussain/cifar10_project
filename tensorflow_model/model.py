# tensorflow_model/model.py
import tensorflow as tf
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('tensorflow_model/cifar10_model.h5')

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def predict(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(32, 32))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0  # Normalize the image

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    prediction_confidences = predictions[0].tolist()
    
    return predicted_class, prediction_confidences
