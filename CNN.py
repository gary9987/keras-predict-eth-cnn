import tensorflow as tf
import pandas as pd
import numpy as np

from solve_cudnn_error import *

solve_cudnn_error()

def make_train(data, k_length = 13):

    x_df = []
    y_df = []
    for i in range(0, data.shape[0]- k_length - 58):
        x_df.append(data[i:i+k_length].to_numpy())
        y_df.append(data['strategy'][i])

    return np.array(x_df), np.array(y_df)

if __name__ == '__main__':  # For test Class


    data = pd.read_csv('output2.csv', dtype=np.float)

    x, y = make_train(data)
    x = x.reshape(43110, 13, 13, 1)
    print(y[:20])
    y = tf.keras.utils.to_categorical(y, 3)
    print(y[:20])
    print(x.shape, y.shape)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu',padding='same', input_shape=(13, 13, 1)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu), # 128
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model.fit(x, y, batch_size = 20, epochs = 10, validation_split=0.8)
    model.save('model.h5')


