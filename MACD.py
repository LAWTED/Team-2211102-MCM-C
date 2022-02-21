from collections import defaultdict
import datetime
import itertools
import time
# import matplotlib.pyplot as plt
import talib as ta
import pandas as pd
import csv
import numpy as np
from datetime import datetime

# calculate wealth of opration
def calWealth(buy, sell,result):
    wealth = 1000
    hold = 0
    buyTime = []
    sellTime = []
    earn = []
    wealthArray = []
    fee = []
    firstBuyDate = datetime.strptime(buy[0], '%m/%d/%y')
    firstSellDate = datetime.strptime(sell[0], '%m/%d/%y')
    # 第一天就买入了
    if firstBuyDate > firstSellDate:
        buy = ['9/11/16'] + buy
    # 判断buyTime and sellTime
    def processBS(buyTime,sellTime):
        # print(buyTime,sellTime)
        bt = buyTime
        st = sellTime
        a = [datetime.strptime(i, '%m/%d/%y') for i in buyTime]
        b = [datetime.strptime(i, '%m/%d/%y') for i in sellTime]
        p = 0
        q = 0
        m = len(a)
        n = len(b)
        pre = a[0]
        flag = 0
        na = [bt[0]]
        nb = []
        while p < m and q < n:
          if flag == 0:
            while  q < n and  b[q] <= pre:
              q += 1
            if q < n:
              pre = b[q]
              nb.append(st[q])
            flag = 1
            continue
          if flag == 1:
            while p < m and  a[p] <= pre:
              p += 1
            if p < m:
              pre = a[p]
              na.append(bt[p])
            flag = 0
        # print(na,nb)
        return (na,nb)
    buy,sell = processBS(buy, sell)
    for i in range(len(buy)):
        # 只有涨了才卖
        if datetime.strptime(buy[i], '%m/%d/%y') < datetime.strptime(sell[i], '%m/%d/%y') and result[buy[i]] < result[sell[i]]:
            hold = wealth / result[buy[i]]
            # 收益 - 买入 * 0.02 - 卖出 * 0.02
            if ((result[sell[i]] - result[buy[i]]) * hold) > (result[buy[i]]+result[sell[i]]) * hold * 0.02 :
                buyTime.append(datetime.strptime(buy[i], '%m/%d/%y'))  # 添加买入时间
                sellTime.append(datetime.strptime(sell[i], '%m/%d/%y'))  # 卖出
                # wealth 增加了 卖出价格 减去 买入价格 乘份额
                wealth += ((result[sell[i]] - result[buy[i]]) * hold) - (result[buy[i]]+result[sell[i]]) * hold * 0.02
                earn.append(hold * (result[sell[i]] - result[buy[i]]))
                fee.append((result[buy[i]]+result[sell[i]]) * hold * 0.02)
                wealthArray.append(wealth)
                # print('buy at '+buy[i]+' sell at ' + sell[i] + ' earn ' + str(
                    # hold * (result[sell[i]] - result[buy[i]])) + ' NOW Wealth ' + str(wealth))
    return (wealth, buyTime, sellTime, earn, wealthArray)

# write to csv the operation
def write2cvs(wealth, buyTime, sellTime, earn, wealthArray):
    dataframe = pd.DataFrame(
        {'buy_time': buyTime, 'sell_time': sellTime, 'earn': earn, 'wealth': wealthArray})
    dataframe.to_csv("operate-%s.csv"%(time.strftime("%m-%d-%H-%M", time.localtime())) , index=False, sep=',')

def writeBuySellCSV(result, buyTime, sellTime):
    daterow = []
    pricerow = []
    buyrow = []
    sellrow = []
    for k,v in result.items():
        daterow.append(datetime.strptime(k, '%m/%d/%y'))
        pricerow.append(v)
        if datetime.strptime(k, '%m/%d/%y') in sellTime:
            sellrow.append(v)
        else:
            sellrow.append(None)
        if datetime.strptime(k, '%m/%d/%y') in buyTime:
            buyrow.append(v)
        else:
            buyrow.append(None)
    for i in range(0, len(result), 166):
        dataframe = pd.DataFrame(
            {'date_time':daterow[i:i+166], 'price': pricerow[i:i+166], 'buy_time': buyrow[i:i+166], 'sell_time': sellrow[i:i+166]})
        dataframe.plot(title='MACD')
        # plt.show()
        dataframe.to_csv("Buy&Sell-%s--%d.csv"%((time.strftime("%m-%d-%H-%M", time.localtime())),i) , index=False, sep=',')
    return

# findBest
def findBest():
    ans = 0
    buy = []
    sell = []
    i = 6
    j = 30
    k = 6
    res = []
    for i in range(2, 30):
        for j in range(10, 100):
            for k in range(2, 30):
                total = MACD(df2, i, j, k, buy, sell)
                buy = []
                sell = []
                f = open("a.txt", 'a')
                f.write("\n %d %d %d %d" %(i, j, k, total) )
                f.close()
                ans = max(ans,total)
                if ans == total:
                    res = [i,j,k]
    print(res,ans)

# create reslut, read csv
def readCSV():
    # 读取csv至字典
    csvFile = open("BCHAIN-MKPRU.csv", "r")
    reader = csv.reader(csvFile)
    f = open("MACD_EveryDay.txt", 'a')

    # 建立空字典
    result = {}
    for item in reader:
        # 忽略第一行
        if reader.line_num == 1:
            continue
        result[item[0]] = float(item[1])

    csvFile.close()
    return result

# true everyday result
def createTRUEResult(result):
    df = []
    buy = []
    sell = []
    # 生成每日历史数据
    for i in range(1,len(result)):
        df.append(dict(itertools.islice(result.items(), i)))

    for i in range(33,len(result)):
        df2 = {
            'date': [],
            'close': []
        }
        for k, v in df[i].items():
                df2['close'].append(float(v))
                df2['date'].append(k)
                # operate(df2)
        buy, sell = MACD_TRUE(df2, buy, sell)
    wealth, buyTime, sellTime, earn, wealthArray = calWealth(buy, sell)
    write2cvs(wealth, buyTime, sellTime, earn, wealthArray)

# simulate everyday result
def createSIMUResult(result):
    df2 = {
        'date': [],
        'close': []
    }
    for k, v in result.iterrows():
        df2['close'].append(float(v['Value']))
        df2['date'].append(k)
    return MACD_SIMU(df2)

# MACD calculation in TRUEMODE
def MACD_TRUE(df2, buy, sell):
    dif, dea, hist = ta.MACD(
        np.array(df2['close']), fastperiod=8, slowperiod=30, signalperiod=13)
    df3 = pd.DataFrame({'dif': dif[33:], 'dea': dea[33:], 'hist': hist[33:]},
                       index=df2['date'][33:], columns=['dif', 'dea', 'hist'])
    # df3.plot(title='MACD')
    # plt.show()
    datenumber = int(df3.shape[0])
    for i in range(datenumber-1):
        if ((df3.iloc[i, 0] <= df3.iloc[i, 1]) & (df3.iloc[i+1, 0] >= df3.iloc[i+1, 1])):
            if df3.index[i+1] == df2['date'][-1]:
                # print('buy ' + df3.index[i+1])
                buy.append(df3.index[i+1])
        if ((df3.iloc[i, 0] >= df3.iloc[i, 1]) & (df3.iloc[i+1, 0] <= df3.iloc[i+1, 1])):
            if df3.index[i+1] == df2['date'][-1]:
                # print('sell ' + df3.index[i+1])
                sell.append(df3.index[i+1])
    return (buy, sell)

# MACD calculation in SIMUMODE
def MACD_SIMU(df2):
    def calMACD(p,d,q):
        buy = []
        sell = []
        # 8 30 13
        dif, dea, hist = ta.MACD(
            np.array(df2['close']), fastperiod=p, slowperiod=d, signalperiod=q)
        df3 = pd.DataFrame({'dif': dif[33:], 'dea': dea[33:], 'hist': hist[33:]},
                        index=df2['date'][33:], columns=['dif', 'dea', 'hist'])
        # df3.plot(title='MACD')
        # plt.show()
        datenumber = int(df3.shape[0])
        for i in range(datenumber-1):
            if ((df3.iloc[i, 0] <= df3.iloc[i, 1]) & (df3.iloc[i+1, 0] >= df3.iloc[i+1, 1])):
                buy.append(df3.index[i+1])
            if ((df3.iloc[i, 0] >= df3.iloc[i, 1]) & (df3.iloc[i+1, 0] <= df3.iloc[i+1, 1])):
                sell.append(df3.index[i+1])
        return (buy,sell)
    buy, sell = calMACD(8,30,13)
    # wealth, buyTime, sellTime, earn, wealthArray = calWealth(buy, sell, result)
    # write2cvs(wealth, buyTime, sellTime, earn, wealthArray)
    return (buy, sell)

def MACD_MAIN(result):
    date = result['Date']
    buyTime, sellTime = createSIMUResult(result)
    Action = [0] * len(result)
    for i in range(len(date)):
        if date[i] in buyTime:
            Action[i] = 1
        if date[i] in sellTime:
            Action[i] = -1
    return Action




if __name__ == '__main__':
    debug = True
    result = readCSV()
    # TRUE MODE
    if (debug == False):
        wealth, buyTime, sellTime, earn, wealthArray, dif, dea, hist = createTRUEResult(result)
        writeBuySellCSV(result, buyTime, sellTime)

    # SIMU MODE
    if (debug == True):
        wealth, buyTime, sellTime, earn, wealthArray, dif, dea, hist = createSIMUResult(result)
        # writeBuySellCSV(result, buyTime, sellTime)


