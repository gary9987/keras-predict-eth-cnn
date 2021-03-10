import requests
import csv
import json

BinanceAPI = "https://fapi.binance.com" # 目前使用合約主網

SystemStateAPI = '/fapi/v1/ping'
req = requests.get(BinanceAPI + SystemStateAPI)
print('Server return status code is', req.status_code)

# function =====================================================================
def getKLine(symbol='ETHUSDT', interval='15m', limit='1000', endTime=0):
    # K線數據 GET /api/v3/klines
    KLineAPI = '/fapi/v1/klines' # 合約主網api
    #KLineAPI = '/api/v3/klines' # 槓桿主網api

    if (endTime == 0):
        my_params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    else:
        my_params = {'symbol': symbol, 'interval': interval, 'limit': limit, 'endTime': endTime}

    req = requests.get(BinanceAPI + KLineAPI, params=my_params)
    kline = json.loads(req.text)

    return kline
'''
[
  [
    1499040000000,      // 开盘时间 0
    "0.01634790",       // 开盘价1
    "0.80000000",       // 最高价2
    "0.01575800",       // 最低价3
    "0.01577100",       // 收盘价(当前K线未结束的即为最新价)4
    "148976.11427815",  // 成交量5
    1499644799999,      // 收盘时间
    "2434.19055334",    // 成交额
    308,                // 成交笔数
    "1756.87402397",    // 主动买入成交量
    "28.46694368",      // 主动买入成交额
    "17928899.62484339" // 请忽略该参数
  ]
]
'''



import datetime

# 開啟輸出的 CSV 檔案

with open('output.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)

    for i in range(1000):
        endTime = 1609459200000 + i * 900000 * 1000  
        dt_object = datetime.datetime.fromtimestamp(endTime / 1000)
        print(dt_object)

        kline = getKLine(endTime = endTime)
        # 建立 CSV 檔寫入器
        
        # 寫入一列資料
        for i in kline:
            writer.writerow([i[0], i[1], i[2], i[3], i[4]], i[5])


    
        
    


