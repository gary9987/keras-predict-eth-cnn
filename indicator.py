import numpy as np
import math
import pandas as pd
import talib
from sklearn.preprocessing import MinMaxScaler

def getMacdHist(data, fast_length=7, slow_length=52, signal_length=8):
    float_data = [float(x) for x in data['close']]
    np_float_data = np.array(float_data)
    macd, signal, hist = talib.MACD(np_float_data, fastperiod=fast_length, slowperiod=slow_length, signalperiod=signal_length)
    return hist

def normolization(data):
    list = []
    # 58開始MACD有資料

    for i in range(58):
        list.append(i)

    data.drop(list, inplace = True)
    data['close'] = data['close'].apply(lambda x: float(x)/3000)
    data['RSI'] = data['RSI'].apply(lambda x: x/100)
    data['K'] = data['K'].apply(lambda x: x/100)
    data['D'] = data['D'].apply(lambda x: x/100)

    # 26.34791967 -37.93686256
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(data['MacdHist'].to_numpy().reshape(-1, 1))
    tmp = scaler.transform(data['MacdHist'].to_numpy().reshape(-1, 1))

    data['MacdHist'] = pd.DataFrame(tmp)

    data['EMA9'] = data['EMA9'].apply(lambda x: x / 3000)
    data['EMA19'] = data['EMA19'].apply(lambda x: x / 3000)
    data['CMO'] = data['CMO'].apply(lambda x: (x+100) / 200)

    # [4536995.426] [-3861686.569]
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(data['OBV'].to_numpy().reshape(-1, 1))
    print(scaler.data_max_, scaler.data_min_)
    tmp = scaler.transform(data['OBV'].to_numpy().reshape(-1, 1))
    data['OBV'] = pd.DataFrame(tmp)

    data['ROC'] = data['ROC'].apply(lambda x: (x+100) / 200)
    data['PPO'] = data['PPO'].apply(lambda x: (x+30) / 60)

    # [607.88206568] [-619.41408855]
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(data['CCI'].to_numpy().reshape(-1, 1))
    print(scaler.data_max_, scaler.data_min_)
    tmp = scaler.transform(data['CCI'].to_numpy().reshape(-1, 1))
    data['CCI'] = pd.DataFrame(tmp)

    return data


if __name__ == '__main__':  # For test Class
    data = pd.read_csv('output.csv', dtype=np.float)
    data['RSI'] = talib.RSI(data['close'], timeperiod = 14)
    data['K'], data['D'] = talib.STOCH(data['high'],
                                                   data['low'],
                                                   data['close'],
                                                   fastk_period=18,
                                                   slowk_period=3,
                                                   slowk_matype=0,
                                                   slowd_period=3,
                                                   slowd_matype=0)
    data['MacdHist'] = getMacdHist(data, 7, 52, 8)
    data['EMA9'] = talib.EMA(data['close'], timeperiod=9)
    data['EMA19'] = talib.EMA(data['close'], timeperiod=19)
    data['CMO'] = talib.stream_CMO(data['close'])
    data['OBV'] = talib.OBV(data['close'], data['volume'])
    data['ROC'] = talib.ROC(data['close'], 10)
    data['PPO'] = talib.PPO(data['close'], 10, 21)
    data['CCI'] = talib.CCI(data['high'], data['low'], data['close'], 20)

    data.drop(columns=['open', 'high', 'low', 'volume'], inplace=True)
    data = normolization(data)
    data.to_csv('output2.csv', index=None)

