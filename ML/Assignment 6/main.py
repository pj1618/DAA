import numpy as np
import tensorflow as tf
tf.keras.models.Model()
df = np.loadtxt('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv',delimiter=',')

#input
x = df[:,:8]

#output
y = df[:,8]

print(y)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

#input layer
model.add(Dense(12, input_dim=8, activation='relu'))

#hidden layers
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))

#output layer
model.add(Dense(1, activation='sigmoid'))

#compile the model
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

#train the model
model.fit(x, y, epochs=100, batch_size=10)

#Evaluate the model
model.evaluate(x,y)

#Summary of the model
model.summary()
