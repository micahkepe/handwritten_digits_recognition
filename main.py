import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load MNIST Dataset and separate training and testing data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Creating the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.save('handwritten.model')
model = tf.keras.models.load_model('handwritten.model')

# Evaluating the model
loss, accuracy = model.evaluate(x_test, y_test)

print("====")
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")
print("====")

# Testing model on created 'digits' dataset
image_number = 1

while os.path.isfile(f"venv/digits/digit{image_number}.png"):
    try:
        print(f"Digit {image_number}")
        img = cv2.imread(f"venv/digits/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"Digit is most likely a {np.argmax(prediction)}")
        plt.imshow(img[0])
        plt.show()
        print("=====")
    except:
        print("Error!")
    finally:
        image_number += 1
