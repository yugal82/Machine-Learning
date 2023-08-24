import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


# we are importing the digits datasets from keras database
# each image of the digit is a 28x28 grid thats is 784 pixels of image. And each pixel has a value from 0 to 255 where 0 represents black and 255 represents white. 
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
# print(len(X_train))

# print(X_train[0].shape) #output - (28, 28)

# print(X_train[0])
# [[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   3  18  18  18 126 136 175  26 166 255 247 127   0   0   0   0]
#  [  0   0   0   0   0   0   0   0  30  36  94 154 170 253 253 253 253 253 225 172 253 242 195  64   0   0   0   0]
#  [  0   0   0   0   0   0   0  49 238 253 253 253 253 253 253 253 253 251 93  82  82  56  39   0   0   0   0   0]
#  [  0   0   0   0   0   0   0  18 219 253 253 253 253 253 198 182 247 241 0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0  80 156 107 253 253 205  11   0  43 154 0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0  14   1 154 253  90   0   0   0   0 0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0 139 253 190   2   0   0   0 0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0  11 190 253  70   0   0   0 0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0  35 241 225 160 108   1 0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0  81 240 253 253 119 25   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0  45 186 253 253 150  27   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  16  93 252 253 187   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 249 253 249  64   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0  46 130 183 253 253 207   2   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0  39 148 229 253 253 253 250 182   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0  24 114 221 253 253 253 253 201 78   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0  23  66 213 253 253 253 253 198  81   2 0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0  18 171 219 253 253 253 253 195  80   9   0   0 0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0  55 172 226 253 253 253 253 244 133  11   0   0   0   0 0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0 136 253 253 253 212 135 132  16   0   0   0   0   0   0 0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 0   0   0   0   0   0   0   0   0   0]]

# plt.gray()
# plt.matshow(X_train[0])
# plt.show()

# print(X_train.shape)
# (60000, 28, 28)

# scaling the data between 0 to 1 range to increase the efficacy of the model
X_train = X_train / 255
X_test = X_test / 255

X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)
# print(X_train_flattened.shape, "\n", X_test_flattened.shape)
# (60000, 784)
# (10000, 784)


# print(X_train_flattened[0])
# [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   3  18  18  18 126 136 175  26 166 255
#  247 127   0   0   0   0   0   0   0   0   0   0   0   0  30  36  94 154
#  170 253 253 253 253 253 225 172 253 242 195  64   0   0   0   0   0   0
#    0   0   0   0   0  49 238 253 253 253 253 253 253 253 253 251  93  82
#   82  56  39   0   0   0   0   0   0   0   0   0   0   0   0  18 219 253
#  253 253 253 253 198 182 247 241   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0  80 156 107 253 253 205  11   0  43 154
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0  14   1 154 253  90   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0   0   0   0 139 253 190   2   0
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0  11 190 253  70   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  35 241
#  225 160 108   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0  81 240 253 253 119  25   0   0   0
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0  45 186 253 253 150  27   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0   0   0   0  16  93 252 253 187
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0 249 253 249  64   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0  46 130 183 253
#  253 207   2   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0  39 148 229 253 253 253 250 182   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0   0   0  24 114 221 253 253 253
#  253 201  78   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0  23  66 213 253 253 253 253 198  81   2   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0  18 171 219 253 253 253 253 195
#   80   9   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#   55 172 226 253 253 253 253 244 133  11   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0 136 253 253 253 212 135 132  16
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#    0   0   0   0   0   0   0   0   0   0]


# this network has only 2 layers, i.e the input layer and the output layer. No middle layers are present
network = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])

network.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

network.fit(X_train_flattened, y_train, epochs=5)

network.evaluate(X_test_flattened, y_test)
#output - 313/313 [==============================] - 1s 2ms/step - loss: 0.2704 - accuracy: 0.9236

y_pred = network.predict(X_test_flattened)
print(y_pred[0])
# [2.7256412e-02 1.5590828e-07 4.5070853e-02 9.5546138e-01 2.4044299e-03
#  1.4880754e-01 1.8350169e-06 9.9982405e-01 9.5411792e-02 7.1380591e-01]


# We get 10 values because our network has 10 output values from 0 to 9. The network predicts the image(digit) is which number. Since there are 10 output values ranging from 0 to 9, the values of the above list corresponds to the 10 values(0-9). The max values corresponding to a number is the prediction(output)
# For eg. y_pred[0] is predicted to be 7 as we can see that the max value of above list is at 7th index
# (9.9982405e-01)

print(np.argmax(y_pred[0]))
# output - 7


# Now this network has a middle layer which was unlike the previous case above. By adding a middle layer in between, the accuracy of the network increases.
# network = keras.Sequential([
#     keras.layers.Dense(100, input_shape=(784,), activation='relu'),
#     keras.layers.Dense(10, activation='sigmoid')
# ])

# network.compile(
#     optimizer='adam', 
#     loss='sparse_categorical_crossentropy', 
#     metrics=['accuracy']
# )
