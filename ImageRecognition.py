import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
'''
x_train2 = []
y_train2 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
for i in range(1, 10):
    img = Image.open(f"FinalProject/Digits/{i}.png")
    img = img.resize((28, 28)).convert('L')
    img_array = np.array(img)
    img_array = np.invert(img_array)
    img_array = tf.keras.utils.normalize(img_array)
    x_train2.append(img_array)

x_train = np.concatenate((x_train, x_train2), axis = 0)
y_train = np.concatenate((y_train, y_train2), axis = 0)    
'''
model.fit(x_train, y_train, epochs=10, verbose=2)

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

for i in range(23, 28):
    prediction = model.predict(x_test[i].reshape(1, 28, 28))
    print(f"This digit is probably a: {np.argmax(prediction)}")
    plt.imshow(x_test[i], cmap='gray')
    plt.show()


for i in range(1, 10):
    img = Image.open(f"FinalProject/Digits/{i}.png")
    img = img.resize((28, 28)).convert('L')
    img_array = np.array(img)
    img_array = np.invert(img_array)
    img_array = tf.keras.utils.normalize(img_array)

    prediction = model.predict(img_array.reshape(1, 28, 28))
    print(f"The uploaded image is predicted to be: {np.argmax(prediction)}")
    
    plt.imshow(img_array, cmap='gray')
    plt.show()

