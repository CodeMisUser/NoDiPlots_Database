# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

class_names = ['Particle (positive)', 'Particle (Negative)', 'Protrusion (positive)', 'Protrusion (negative)']

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(25, 25)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(4)
])

# Compile the model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Model summary
model.summary()


fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print("shit")

