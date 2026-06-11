import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([5,8,13,20,29,40,53,68,85,104])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(33,activation='relu',input_shape=[1]),
    
    tf.keras.layers.Dense(1)
])
my_optymizer = tf.keras.optimizers.Adam(learning_rate=0.82)
model.compile(
    optimizer = my_optymizer,
    loss = 'mse',
    metrics = ['accuracy']
)
model.fit(x,y,epochs=300)
x_pred = np.array([11])
m = model.predict(x_pred)
print(m)
y_pred = model.predict(x)
plt.plot(x,y_pred)
plt.plot(x,y)
plt.show()