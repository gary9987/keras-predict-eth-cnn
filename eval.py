import tensorflow as tf
import pandas as pd
import numpy as np
import train
from sklearn.metrics import classification_report

if __name__ == '__main__':

    window = 50

    eval_data = pd.read_csv('1-3_indicatored.csv', dtype=np.float)
    XEval, YEval = train.make_train(eval_data, window)
    XEval = np.reshape(XEval, (XEval.shape[0], XEval.shape[1], XEval.shape[2], 1))
    YEval = tf.keras.utils.to_categorical(YEval, 3)
    print(XEval.shape, YEval.shape)

    model = tf.keras.models.load_model('model.h5')
    cost = model.evaluate(XEval, YEval, batch_size=100)
    print("test cost: {}".format(cost))

    Y_test = np.argmax(YEval, axis=1)  # Convert one-hot to index
    y_pred = model.predict_classes(XEval)
    print(classification_report(Y_test, y_pred))


