def Labelling(data):
    windowSize = 11
    counterRow = 0
    result = [-2, -2, -2, -2, -2]
    while(counterRow < len(data.index)):
        min = 1000000000000
        max = 0
        counterRow += 1
        if(counterRow >= windowSize):
            windowBeginIndex = counterRow - windowSize
            windowEndIndex = windowBeginIndex + windowSize - 1
            windowMiddleIndex = (windowBeginIndex + windowEndIndex) / 2
            for i in range(windowBeginIndex, windowEndIndex+1) :
                number = data['close'][i]
                if (number < min):
                    min = number
                    minIndex = i
                if (number > max):
                    max = number
                    maxIndex = i

            if (maxIndex == windowMiddleIndex):
                result.append(-1) # Sell
            elif (minIndex == windowMiddleIndex):
                result.append(1) # Buy
            else:
                result.append(0) # Hold
    
    result.append(-2)
    result.append(-2)
    result.append(-2)
    result.append(-2)
    result.append(-2)
    return result
    
import pandas as pd
df = pd.read_csv('output.csv')
data = Labelling(df)
df['stratrgy'] = data
df.to_csv('output.csv', index=None)