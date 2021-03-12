import tensorflow as tf
import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from solve_cudnn_error import *
from sklearn.model_selection import train_test_split

solve_cudnn_error()

def make_train(data, k_length = 13):

    x_df = []
    y_df = []

    for i in range(0, data.shape[0]- k_length - 58):
        y_df.append(data['strategy'][i])

    data.drop(columns=['strategy'], inplace=True)

    for i in range(0, data.shape[0]- k_length - 58):
        x_df.append(np.rot90(data[i:i+k_length].to_numpy(), 1))

    return np.array(x_df), np.array(y_df)

if __name__ == '__main__':  # For test Class

    window = 16
    data = pd.read_csv('output2.csv', dtype=np.float)

    x, y = make_train(data, window)
    # 43110 13 13
    orig_shape = x.shape
    print(orig_shape)
    print(x.shape)

    x = np.reshape(x, (x.shape[0], x.shape[2]*x.shape[1]))
    print(x.shape)

    X_new, Y_new = RandomUnderSampler().fit_resample(x, y)


    X_new = np.reshape(X_new, (X_new.shape[0], orig_shape[1], orig_shape[2], 1))
    Y_new = tf.keras.utils.to_categorical(Y_new, 3)
    print(X_new.shape, Y_new.shape)

    XTraining, XValidation, YTraining, YValidation = train_test_split(X_new, Y_new, stratify=Y_new, test_size=0.2, random_state = 7 )  # before model building



    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu',padding='same', input_shape=(orig_shape[1], orig_shape[2], 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation=tf.nn.relu), # 128
        tf.keras.layers.Dense(3, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    #model.fit(XTraining, YTraining, batch_size=20, epochs=100, validation_data=(XValidation, YValidation))
    model.fit(XTraining, YTraining, batch_size=20, epochs=60, validation_split = 0.15)

    cost = model.evaluate(XValidation, YValidation, batch_size=20)
    print("test cost: {}".format(cost))
    #model.save('model.h5')


