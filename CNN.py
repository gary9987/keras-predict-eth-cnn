import tensorflow as tf
import pandas as pd
import numpy as np


def make_train(data, k_length = 15):
    x_df = []
    y_df = []
    for i in range(data.shape[0]- k_length - 58):
        x_df.append(data[i:i+k_length])
        y_df.append(data['strategy'][i])

    return np.array(x_df), np.array(y_df)

if __name__ == '__main__':  # For test Class


    data = pd.read_csv('output2.csv')
    data.drop(columns = ['time', 'open', 'high', 'low', 'volume'], inplace=True)
    x, y = make_train(data)
    x = x.reshape(-1, 15, 13, 1)
    print(x.shape, y.shape)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(15, 13, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(128, activation=tf.nn.relu), # 128
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    history = model.fit(x, y, batch_size = 20, epochs = 300, validation_split=0.3)


