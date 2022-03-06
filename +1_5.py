# -*- coding: utf-8 -*-
"""+1.5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rzcHycJ8c3V3Zz6ccVPE8dklLVl77Dov
"""

# Proof of IA

def calc(n):
  return n + 1

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

input = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], dtype=int)
output = np.array([2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5], dtype=float)

nu = [20, 16, 1]

model = tf.keras.Sequential()

l1 = model.add(tf.keras.layers.Dense(units=nu[0], input_shape=[1])) # l1

l2 = model.add(tf.keras.layers.Dense(units=nu[1], input_shape=[1])) # l2

l3 = model.add(tf.keras.layers.Dense(units=nu[2], input_shape=[1])) # l3

model.compile(
     optimizer = tf.optimizers.Adam(0.1),
     loss="mean_squared_error",
)

print("Entrenando ...")
historial = model.fit(input, output, epochs=1500, verbose=0)
print("Entrenado.")

plt.xlabel("Epoca")
plt.ylabel("Magnitud de perdida")
plt.plot(historial.history["loss"])

print(model.predict([99999]))