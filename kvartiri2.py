import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

feature_columns = None

def df_maker(name, train=True):
    global scaler_features
    global scaler_target
    global feature_columns

    df = pd.read_csv(name, decimal=',')

    target_cols = 'Цена_квартиры'
    feature_cols = ['Район','Тип_квартиры','Площадь_общая','Количество_комнат','Этаж','Год_постройки','Ближайшее_метро_мин','Парковка','Ремонт_от_застройщика']

    df = df[[target_cols] + feature_cols]
    df = df.dropna()

    df['Парковка'] = df['Парковка'].map({'Да': 1, 'Нет': 0})
    df['Ремонт_от_застройщика'] = df['Ремонт_от_застройщика'].map({'Да': 1, 'Нет': 0})

    df = pd.get_dummies(df, columns=['Район'])
    df = pd.get_dummies(df, columns=['Тип_квартиры'])

    feature_cols_to_scale = ['Площадь_общая','Количество_комнат','Этаж','Год_постройки','Ближайшее_метро_мин']

    if train:
        df[feature_cols_to_scale] = scaler_features.fit_transform(df[feature_cols_to_scale])
        df[target_cols] = scaler_target.fit_transform(df[[target_cols]])
        feature_columns = df.drop(target_cols, axis=1).columns
    else:
        df = df.reindex(columns=list(feature_columns) + [target_cols], fill_value=0)
        df[feature_cols_to_scale] = scaler_features.transform(df[feature_cols_to_scale])

    X_scaled = df.drop(target_cols, axis=1)
    y = df[target_cols]

    return X_scaled, y

X_sc, y = df_maker('kvartiri.csv')

X_train, X_test, y_train, y_test = train_test_split(
    X_sc, y, test_size=0.2, random_state=42
)

print(X_train)

model = keras.Sequential([
    keras.layers.Dense(70, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.fit(X_train, y_train, epochs=300, validation_data=(X_test, y_test), verbose=1)

X_test, y_t = df_maker('testkv.csv', train=False)

print(X_test)

predictions_scaled = model.predict(X_test)

print(predictions_scaled)

predictions_real = scaler_target.inverse_transform(predictions_scaled)

print("\nПредсказанные значения:")

for i in range(len(predictions_real)):
    print(f"Предсказание: {predictions_real[i][0]:.0f} ₽")