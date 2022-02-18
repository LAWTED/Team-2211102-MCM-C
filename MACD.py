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



# data 日期 close 收盘价
buy = []
sell = []
# fw = open("./query_deal.txt", 'w')

df2 = {
    'date': [],
    'close': []
}
for k, v in result.items():
    df2['close'].append(float(v))
    df2['date'].append(k)

def MACD(df2):
    dif, dea, hist = ta.MACD(
        np.array(df2['close']), fastperiod=2, slowperiod=26, signalperiod=2)
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
    return(dif, dea, hist)

# df = []
# # 生成每日历史数据
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

MACD(df2)
buyTime = []
sellTime = []
earn = []
wealthArray = []


def calWealth(buy, sell):
    wealth = 1000
    hold = 0

    firstBuyDate = datetime.strptime(buy[0], '%m/%d/%y')
    firstSellDate = datetime.strptime(sell[0], '%m/%d/%y')
    # 第一天就买入了
    if firstBuyDate > firstSellDate:
        buy = ['9/11/16'] + buy
    for i in range(len(buy)):
        # 只有涨了才卖
        if result[buy[i]] < result[sell[i]]:
            hold = wealth / result[buy[i]]
            # 收益 - 买入 * 0.02 - 卖出 * 0.02
            if ((result[sell[i]] - result[buy[i]]) * hold) > (result[buy[i]]+result[sell[i]]) * hold * 0.02 :
                buyTime.append(datetime.strptime(buy[i], '%m/%d/%y'))  # 添加买入时间
                sellTime.append(datetime.strptime(sell[i], '%m/%d/%y'))  # 卖出
                # wealth 增加了 卖出价格 减去 买入价格 乘份额
                wealth += ((result[sell[i]] - result[buy[i]]) * hold) - (result[buy[i]]+result[sell[i]]) * hold * 0.02
                earn.append(hold * (result[sell[i]] - result[buy[i]]))
                wealthArray.append(wealth)
                # print('buy at '+buy[i]+' sell at ' + sell[i] + ' earn ' + str(
                    # hold * (result[sell[i]] - result[buy[i]])) + ' NOW Wealth ' + str(wealth))
    print(len(buyTime))
    return wealth
total = calWealth(buy,sell)
print(total)


def write2cvs(buyTime, sellTime, earn, wealthArray):
    dataframe = pd.DataFrame(
        {'buy_time': buyTime, 'sell_time': sellTime, 'earn': earn, 'wealth': wealthArray})
    dataframe.to_csv("operate.csv", index=False, sep=',')

# write2cvs(buyTime, sellTime, earn, wealthArray)





# def findBest():

#     ans = 0
#     buy = []
#     sell = []
#     i = 6
#     j = 30
#     k = 6
#     res = []
#     for i in range(2, 30):
#         for j in range(10, 100):
#             for k in range(2, 30):
#                 total = MACD(df2, i, j, k, buy, sell)
#                 buy = []
#                 sell = []
#                 f = open("a.txt", 'a')
#                 f.write("\n %d %d %d %d" %(i, j, k, total) )
#                 f.close()
#                 ans = max(ans,total)
#                 if ans == total:
#                     res = [i,j,k]
#     print(res,ans)

# findBest()
