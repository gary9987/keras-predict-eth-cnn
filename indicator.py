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
    data['MFI'] = data['MFI'].apply(lambda x: x / 100)
    data['WILLR'] = data['WILLR'].apply(lambda x: -x / 100)
    # 26.34791967 -37.93686256
    '''
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(data['MacdHist'].to_numpy().reshape(-1, 1))
    tmp = scaler.transform(data['MacdHist'].to_numpy().reshape(-1, 1))
    '''
    data['MacdHist'] = data['MacdHist'].apply(lambda x: (x + 50) / 100 )
    data['SMA'] = data['SMA'].apply(lambda x: x / 3000)
    data['TEMA'] = data['TEMA'].apply(lambda x: x / 3000)
    data['EMA9'] = data['EMA9'].apply(lambda x: x / 3000)
    data['WMA'] = data['WMA'].apply(lambda x: x / 3000)
    data['CMO'] = data['CMO'].apply(lambda x: (x+100) / 200)

    # [4536995.426] [-3861686.569]
    '''
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(data['OBV'].to_numpy().reshape(-1, 1))
    tmp = scaler.transform(data['OBV'].to_numpy().reshape(-1, 1))
    '''
    data['OBV'] = data['OBV'].apply(lambda x: (x + 5000000) / 10000000)

    data['ROC'] = data['ROC'].apply(lambda x: (x+100) / 200)
    data['PPO'] = data['PPO'].apply(lambda x: (x+30) / 60)

    # [607.88206568] [-619.41408855]
    '''
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(data['CCI'].to_numpy().reshape(-1, 1))
    print(scaler.data_max_, scaler.data_min_)
    tmp = scaler.transform(data['CCI'].to_numpy().reshape(-1, 1))
    '''
    data['CCI'] = data['CCI'].apply(lambda x: (x + 700) / 1400)

    return data


if __name__ == '__main__':  # For test Class
    data = pd.read_csv('eval_labeled.csv', dtype=np.float)
    data.drop(columns=['time'], inplace=True)
    data['RSI'] = talib.RSI(data['close'], timeperiod = 14)
    data['WILLR'] = talib.WILLR(data['high'], data['low'], data['close'])
    data['K'], data['D'] = talib.STOCH(data['high'],
                                                   data['low'],
                                                   data['close'],
                                                   fastk_period=18,
                                                   slowk_period=3,
                                                   slowk_matype=0,
                                                   slowd_period=3,
                                                   slowd_matype=0)
    data['WMA'] = talib.WMA(data['close'], 20)
    data['EMA9'] = talib.EMA(data['close'], timeperiod=9)
    data['SMA'] = talib.SMA(data['close'], timeperiod=18)
    data['TEMA'] = talib.TEMA(data['close'], timeperiod=15)
    data['MacdHist'] = getMacdHist(data, 7, 52, 8)
    data['CCI'] = talib.CCI(data['high'], data['low'], data['close'], 20)
    data['CMO'] = talib.CMO(data['close'])
    data['PPO'] = talib.PPO(data['close'], 10, 21)
    data['ROC'] = talib.ROC(data['close'], 10)
    data['MFI'] = talib.MFI(data['high'], data['low'], data['close'], data['volume'])
    data['OBV'] = talib.OBV(data['close'], data['volume'])



    data.drop(columns=['open', 'high', 'low', 'volume'], inplace=True)
    data = normolization(data)
    data.to_csv('eval_indicatored.csv', index=None)

