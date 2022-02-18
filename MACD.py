from collections import defaultdict
import datetime
from email.policy import default
import itertools
from typing import List
import matplotlib.pyplot as plt
import talib as ta
import pandas as pd
import baostock as bs
import csv
import numpy as np
from datetime import datetime
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

# 使用talib计算MACD的参数
short_win = 12    # 短期EMA平滑天数
long_win = 26    # 长期EMA平滑天数
macd_win = 20     # DEA线平滑天数

# data 日期 close 收盘价
# buy = []
# sell = []
# fw = open("./query_deal.txt", 'w')


def MACD(df2, fast, slow, signal, buy, sell):
    dif, dea, hist = ta.MACD(
        np.array(df2['close']), fastperiod=fast, slowperiod=slow, signalperiod=signal)
    df3 = pd.DataFrame({'dif': dif[33:], 'dea': dea[33:], 'hist': hist[33:]},
                       index=df2['date'][33:], columns=['dif', 'dea', 'hist'])
    # df3.plot(title='MACD')
    # plt.show()
    # 寻找MACD金叉和死叉
    datenumber = int(df3.shape[0])
    for i in range(datenumber-1):
        if ((df3.iloc[i, 0] <= df3.iloc[i, 1]) & (df3.iloc[i+1, 0] >= df3.iloc[i+1, 1])):
            # print("MACD金叉的日期："+df3.index[i+1])
            buy.append(df3.index[i+1])
            if df3.index[i+1] == df2['date'][-1]:
                # print('buy ' + df3.index[i+1])
                buy.append(df3.index[i+1])
        if ((df3.iloc[i, 0] >= df3.iloc[i, 1]) & (df3.iloc[i+1, 0] <= df3.iloc[i+1, 1])):
            # print("MACD死叉的日期："+df3.index[i+1])
            sell.append(df3.index[i+1])
            # if df3.index[i+1] == df2['date'][-1]:
                # print('sell ' + df3.index[i+1])
    # print(buy, sell)
    total = calWealth(buy, sell)
    # return(dif, dea, hist, buy, sell)
    return total

# df = []
# 生成每日历史数据
# for i in range(1,len(result)):
#     df.append(dict(itertools.islice(result.items(), i)))

# for i in range(33,len(result)):
#     df2 = {
#         'date': [],
#         'close': []
#     }
#     for k, v in df[i].items():
#             df2['close'].append(float(v))
#             df2['date'].append(k)

#     MACD(df2)


buyTime = []
sellTime = []
earn = []
wealthArray = []


def calWealth(buy, sell):
    wealth = 1000
    hold = 0
    # print(len(buy))
    # print(len(sell))
    firstBuyDate = datetime.strptime(buy[0], '%m/%d/%y')
    firstSellDate = datetime.strptime(sell[0], '%m/%d/%y')
    # print(buy, sell)
    # 第一天就买入了
    if firstBuyDate > firstSellDate:
        buy = ['9/11/16'] + buy

        # buy 98个日期 sell 98个日期
    for i in range(len(buy)):
        # 当前的钱除价格算 份额
        # 只有涨了才卖
        if result[buy[i]] < result[sell[i]]:
            buyTime.append(datetime.strptime(buy[i], '%m/%d/%y'))  # 添加买入时间
            sellTime.append(datetime.strptime(sell[i], '%m/%d/%y'))  # 卖出
            hold = wealth / result[buy[i]]
            # wealth 增加了 卖出价格 减去 买入价格 乘份额
            wealth += hold * (result[sell[i]] - result[buy[i]])
            earn.append(hold * (result[sell[i]] - result[buy[i]]))
            wealthArray.append(wealth)
            # print('buy at '+buy[i]+' sell at ' + sell[i] + ' earn ' + str(
                # hold * (result[sell[i]] - result[buy[i]])) + ' NOW Wealth ' + str(wealth))

    return wealth
# total = calWealth(buy,sell)
# print(total)


def write2cvs(buyTime, sellTime, earn, wealthArray):
    dataframe = pd.DataFrame(
        {'buy_time': buyTime, 'sell_time': sellTime, 'earn': earn, 'wealth': wealthArray})
    dataframe.to_csv("operate.csv", index=False, sep=',')

# write2cvs(buyTime, sellTime, earn, wealthArray)


df2 = {
    'date': [],
    'close': []
}
for k, v in result.items():
    df2['close'].append(float(v))
    df2['date'].append(k)


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

findBest()
