# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

class_names = ['Particle', 'None']

model_Q = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(25, 25)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2)
])

# Compile the model
model_Q.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Model summary
model.summary()



