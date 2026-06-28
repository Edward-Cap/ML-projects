import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

texts = [
    "Мне очень понравился этот фильм",
"Отличный сервис, быстро и вежливо",
"Качество товара превзошло ожидания",
"Я доволен покупкой",
"Прекрасная работа команды",
"Этот продукт действительно полезен",
"Всё прошло замечательно",
"Очень удобное приложение",
"Хорошая идея и хорошее исполнение",
"С удовольствием буду рекомендовать",
"Фильм оказался скучным и затянутым",
"Сервис ужасный, пришлось долго ждать",
"Товар сломался через день",
"Я разочарован покупкой",
"Плохая поддержка клиентов",
"Приложение постоянно зависает",
"Ничего полезного в этом нет",
"Очень неудобный интерфейс",
"Идея хорошая, но реализация слабая",
"Не советую это покупать",
"Мне понравилось обслуживание",
"Это было очень полезно",
"Все работает отлично",
"Покупка меня порадовала",
"Очень приятный результат",
"Я недоволен качеством",
"Это приложение бесполезно",
"Слишком медленная работа",
"Плохой опыт использования",
"Совсем не рекомендую"
]

labels = np.array([1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0])

tokenizer = Tokenizer(num_words=1000) 
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, padding='post')

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=1000, output_dim=16), 
    tf.keras.layers.Dropout(0.32),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

my_optymizer = tf.keras.optimizers.Adam(learning_rate=0.0019)
model.compile(
    optimizer=my_optymizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(padded, labels, epochs=200)

test_text = [
             "замечательное и удобное приложение"]
test_seq = tokenizer.texts_to_sequences(test_text)     
test_pad = pad_sequences(test_seq, maxlen=padded.shape[1], padding='post')
prediction = model.predict(test_pad)

print("Вероятность положительного отзыва:", prediction)
if prediction >= 0.5:
    print('отзыв положительный')
else:
    print('отзыв отрицательный')

# отзыв должен быть по структуре похож на отзывы из обучения и содержать слова,которые были в обучении. Иначе ответ будет неверным.