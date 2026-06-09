import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

model = tf.keras.models.load_model('model.keras')

def predict_my_digit(image_path):
    img = Image.open(image_path).convert('L').resize((28,28))
    img = ImageOps.invert(img)

    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1,28,28,1)

    prediction = model.predict(img_array)
    result = np.argmax(prediction)
    confidence = np.max(prediction)
    return result,confidence, img

digit, prob, processed_img = predict_my_digit('тройка.jpg')
print(f'результат:{digit}')
print(f'уверенность:{prob*100:.2f}%')
plt.imshow(processed_img,cmap='gray')
plt.show()
