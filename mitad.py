

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

input = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40], dtype=int)
output = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], dtype=int)

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
historial = model.fit(input, output, epochs=1000, verbose=False)
print("Entrenado")

plt.xlabel("Epoca")
plt.ylabel("Magnitud de perdida")
plt.plot(historial.history["loss"])

print(model.predict([24664])[0][0])