import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape =(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=12, validation_data=(X_test,y_test))
# Смотрим общую точность на тестовых данных
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Точность модели: {accuracy*100:.2f}%")

# Делаем прогноз для случайной картинки
prediction = model.predict(X_test)
index = 7 # Попробуй поменять индекс!
print(f"Модель считает, что это: {np.argmax(prediction[index])}")
print(f"Правильный ответ: {y_test[index]}")

plt.imshow(X_test[index].reshape(28,28), cmap='gray')
plt.show()