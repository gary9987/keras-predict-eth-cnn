import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

def tripleBarrier(price, ub, lb, max_period):
    def end_price(s):
        return np.append(s[(s / s[0] > ub) | (s / s[0] < lb)], s[-1])[0] / s[0]

    r = np.array(range(max_period))

    def end_time(s):
        return np.append(r[(s / s[0] > ub) | (s / s[0] < lb)], max_period - 1)[0]

    p = price.rolling(max_period).apply(end_price, raw=True).shift(-max_period + 1)
    t = price.rolling(max_period).apply(end_time, raw=True).shift(-max_period + 1)
    t = pd.Series([t.index[int(k + i)] if not math.isnan(k + i) else np.datetime64('NaT')
                   for i, k in enumerate(t)], index=t.index).dropna()

    signal = pd.Series(0, p.index)
    signal.loc[p > ub] = 1
    signal.loc[p < lb] = -1
    ret = pd.DataFrame({'triple_barrier_profit': p, 'triple_barrier_sell_time': t, 'triple_barrier_signal': signal})

    return ret

    
if __name__ == '__main__':  # For test Class
    df = pd.read_csv('kline.csv')

    data = tripleBarrier(df['close'], 1.06, 0.97, 160)
    print(data)
    df['strategy'] = data['triple_barrier_signal']
    print(df['strategy'].value_counts())
    df.to_csv('output.csv', index=None)
    '''
    new = pd.DataFrame()
    new['stratrgy'] = df['stratrgy']
    new['close'] = df['close']
    new.plot()
    plt.show()
    '''