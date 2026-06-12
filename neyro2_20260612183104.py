import tensorflow as tf
import numpy as np


dollars = np.array([1,2,3,4,5], dtype=float)

rub = np.array([75,150,225,300,375], dtype=float)
model = tf.keras.Sequential([
    tf.keras.layers.Dence(units = 10,activation = 'relu', input_shape = [1]),

    tf.keras.layers.Dence(units = 1)    
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),loss = 'mean_squared_error')
print('тренируем нейросеть рисовать кривые линии...')
model.fit(dollars,rub,epochs = 500, verbose = 0)
print('готово!')

tdata = np.array([6.0])
print(model.predict(tdata))
# model.save('upgrade_v1.keras')
# print('модель сохранена! ищите в папке слева')

