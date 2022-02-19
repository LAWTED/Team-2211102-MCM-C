import operator
from re import A
import sys
from collections import defaultdict
import datetime
from email.policy import default
import enum
import itertools
import time
from typing import List
from venv import create
import matplotlib.pyplot as plt
import talib as ta
import pandas as pd
import baostock as bs
import csv
import numpy as np
from datetime import datetime


# def DP_OP(df2):
#     prices = df2['close']
#     date = df2['date']
#     n = len(prices)
#     dp = [[0, -prices[0]]] + [[0, 0] for _ in range(n - 1)]
#     buyTime = []
#     sellTime = []
#     for i in range(1, n):
#         dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
#         # 卖出
#         if (dp[i][0] == dp[i-1][1] + prices[i]):
#             buyTime.append(date[i])
#         dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
#         # 买入
#         if (dp[i][1] == dp[i-1][0] - prices[i]):
#             sellTime.append(date[i])
#     writeBuySellCSV(result, buyTime, sellTime)
#     # print(buyTime, sellTime)
#     return dp[n - 1][0]

# def writeBuySellCSV(result, buyTime, sellTime):
#     daterow = []
#     pricerow = []
#     buyrow = []
#     sellrow = []
#     for k,v in result.items():
#         daterow.append(datetime.strptime(k, '%m/%d/%y'))
#         pricerow.append(v)
#         if k in sellTime:
#             sellrow.append(v)
#         else:
#             sellrow.append(None)
#         if k in buyTime:
#             buyrow.append(v)
#         else:
#             buyrow.append(None)
#     for i in range(0, len(result), 166):
#         dataframe = pd.DataFrame(
#             {'date_time':daterow[i:i+166], 'price': pricerow[i:i+166], 'buy_time': buyrow[i:i+166], 'sell_time': sellrow[i:i+166]})
#         dataframe.plot(title='MACD')
#         # plt.show()
#         dataframe.to_csv("Buy&Sell-%s--%d.csv"%((time.strftime("%m-%d-%H-%M", time.localtime())),i) , index=False, sep=',')
#     return

# if __name__ == '__main__':
#     result = readCSV()
#     total = createSIMUResult(result)
#     print(total)


# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ optimal_stock_action.py ]
#   Synopsis     [ Best Time to Buy and Sell Stock with Transaction Fee - with Dynamic Programming ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############


def find_optimal_action(priceVec, transFeeRate, use_DP=True):

    _BUY = 1
    _HOLD = 0
    _SELL = -1

    dataLen = len(priceVec)
    actionVec = np.zeros(dataLen)

    # Dynamic Programming method
    if use_DP:
        capital = 1000
        money = [{'money': 0, 'from': 0} for _ in range(dataLen)]
        stock = [{'stock': 0, 'from': 1} for _ in range(dataLen)]

        # DP initialization
        money[0]['money'] = capital
        stock[0]['stock'] = capital * (1 - transFeeRate) / priceVec[0]

        # DP recursion
        for t in range(1, dataLen):

            # find optimal for sell at time t:
            hold = money[t - 1]['money']
            sell = stock[t - 1]['stock'] * priceVec[t] * (1 - transFeeRate)

            if hold > sell:
                money[t]['money'] = hold
                money[t]['from'] = 0
            else:
                money[t]['money'] = sell
                money[t]['from'] = 1

            # find optimal for buy at time t:
            hold = stock[t - 1]['stock']
            buy = money[t - 1]['money'] * (1 - transFeeRate) / priceVec[t]

            if hold > buy:
                stock[t]['stock'] = hold
                stock[t]['from'] = 1
            else:
                stock[t]['stock'] = buy
                stock[t]['from'] = 0

        # must sell at T
        prev = 0
        actionVec[-1] = _SELL

        # DP trace back
        record = [money, stock]
        for t in reversed(range(1, dataLen)):
            prev = record[prev][t]['from']
            actionVec[t - 1] = _SELL if prev == 0 else _BUY

        # Action smoothing
        prevAction = actionVec[0]
        for t in range(1, dataLen):
            if actionVec[t] == prevAction:
                actionVec[t] = _HOLD
            elif actionVec[t] == -prevAction:
                prevAction = actionVec[t]

        return actionVec

    # Baseline method
    else:
        conCount = 3
        for ic in range(dataLen):
            if ic + conCount + 1 > dataLen:
                continue
            if all(x > 0 for x in list(map(operator.sub, priceVec[ic+1:ic+1+conCount], priceVec[ic:ic+conCount]))):
                actionVec[ic] = _BUY
            if all(x < 0 for x in list(map(operator.sub, priceVec[ic+1:ic+1+conCount], priceVec[ic:ic+conCount]))):
                actionVec[ic] = _SELL
        prevAction = _SELL

        for ic in range(dataLen):
            if actionVec[ic] == prevAction:
                actionVec[ic] = _HOLD
            elif actionVec[ic] == -prevAction:
                prevAction = actionVec[ic]
        return actionVec


def writeBuySellCSV(result, buyTime, sellTime):
    daterow = []
    pricerow = []
    buyrow = []
    sellrow = []
    for k, v in result.items():
        daterow.append(datetime.strptime(k, '%m/%d/%y'))
        pricerow.append(v)
        if k in sellTime:
            sellrow.append(v)
        else:
            sellrow.append(None)
        if k in buyTime:
            buyrow.append(v)
        else:
            buyrow.append(None)
    for i in range(0, len(result), 166):
        dataframe = pd.DataFrame(
            {'date_time': daterow[i:i+166], 'price': pricerow[i:i+166], 'buy_time': buyrow[i:i+166], 'sell_time': sellrow[i:i+166]})
        dataframe.plot(title='MACD')
        # plt.show()
        dataframe.to_csv("Buy&Sell-%s--%d.csv" %
                         ((time.strftime("%m-%d-%H-%M", time.localtime())), i), index=False, sep=',')
    return


def profit_estimate(priceVec, transFeeRate, actionVec):

    capital = 1
    capitalOrig = capital
    dataCount = len(priceVec)
    suggestedAction = actionVec

    stockHolding = np.zeros((dataCount))
    total = np.zeros((dataCount))

    total[0] = capital

    for ic in range(dataCount):
        currPrice = priceVec[ic]
        if ic > 0:
            stockHolding[ic] = stockHolding[ic-1]
        if suggestedAction[ic] == 1:
            if stockHolding[ic] == 0:
                stockHolding[ic] = capital * (1 - transFeeRate) / currPrice
                capital = 0

        elif suggestedAction[ic] == -1:
            if stockHolding[ic] > 0:
                capital = stockHolding[ic] * currPrice * (1 - transFeeRate)
                stockHolding[ic] = 0

        elif suggestedAction[ic] == 0:
            pass
        else:
            assert False
        total[ic] = capital + stockHolding[ic] * currPrice * (1 - transFeeRate)
    returnRate = (total[-1] - capitalOrig) / capitalOrig
    return returnRate


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


def createSIMUResult(result):
    df2 = {
        'date': [],
        'close': []
    }
    for k, v in result.items():
        df2['close'].append(float(v))
        df2['date'].append(k)
    return df2


def transBS(result, date, actionVec):
    buyTime = []
    sellTime = []
    for ind, v in enumerate(actionVec):
        if v == 1:
            buyTime.append(date[ind])
        if v == -1:
            sellTime.append(date[ind])
    return (buyTime, sellTime)


def calWealth(buy, sell):
    wealth = 1000
    hold = 0
    buyTime = []
    sellTime = []
    earn = []
    wealthArray = []
    firstBuyDate = datetime.strptime(buy[0], '%m/%d/%y')
    firstSellDate = datetime.strptime(sell[0], '%m/%d/%y')
    fee = []
    # 第一天就买入了
    if firstBuyDate > firstSellDate:
        buy = ['9/11/16'] + buy
    for i in range(len(buy)):
        # 只有涨了才卖
        if result[buy[i]] < result[sell[i]]:
            hold = wealth / result[buy[i]]
            # 收益 - 买入 * 0.02 - 卖出 * 0.02
            if ((result[sell[i]] - result[buy[i]]) * hold) > (result[buy[i]]+result[sell[i]]) * hold * 0.02:
                buyTime.append(datetime.strptime(buy[i], '%m/%d/%y'))  # 添加买入时间
                sellTime.append(datetime.strptime(sell[i], '%m/%d/%y'))  # 卖出
                # wealth 增加了 卖出价格 减去 买入价格 乘份额
                wealth += ((result[sell[i]] - result[buy[i]]) * hold) - \
                    (result[buy[i]]+result[sell[i]]) * hold * 0.02
                earn.append(hold * (result[sell[i]] - result[buy[i]]))
                fee.append((result[buy[i]]+result[sell[i]]) * hold * 0.02)
                wealthArray.append(wealth)
                # print('buy at '+buy[i]+' sell at ' + sell[i] + ' earn ' + str(
                # hold * (result[sell[i]] - result[buy[i]])) + ' NOW Wealth ' + str(wealth))
    return (wealth, buyTime, sellTime, earn, wealthArray, fee)


def write2cvs(result, buyTime, sellTime, earn, wealthArray, fee):
    buy_price = []
    sell_price = []
    diff_price = []
    S_buyTime = []
    S_sellTime = []
    share = []
    priceVec = createSIMUResult(result)['close']
    date = createSIMUResult(result)['date']
    for i in buyTime:
        buy_price.append(result[i])
    for i in sellTime:
        sell_price.append(result[i])
    for i in range(len(buyTime)):
        diff_price.append(sell_price[i]-buy_price[i])
    wealth_before = [1000] + wealthArray[:-1]
    for i in range(len(buyTime)):
        share.append(wealth_before[i]/buy_price[i])
    for i in buyTime:
        S_buyTime.append(datetime.strptime(i, '%m/%d/%y'))
    for i in sellTime:
        S_sellTime.append(datetime.strptime(i, '%m/%d/%y'))
    dataframe = pd.DataFrame({'buy_time': S_buyTime, 'buy_price': buy_price, 'sell_time': S_sellTime, 'sell_price': sell_price, 'wealth_before': wealth_before, 'diff_price': diff_price, 'share': share,'earn': earn, 'fee': fee, 'wealth': wealthArray})

    dataframe.to_csv("operate-%s.csv" % (time.strftime("%m-%d-%H-%M",
                     time.localtime())), index=False, sep=',')


def SDRes(result):
    newRes = []
    for k, v in result.items():
        tmp = {
            'date': datetime.strptime(k, '%m/%d/%y'),
            'close': float(v)
        }
        newRes.append(tmp)
    return newRes

def plotBUYSELL(date, priceVec, actionVec):
    actionVec[0] = 0
    dataframe2 = pd.DataFrame({'date': date,'price': priceVec})
    fig = dataframe2.plot(title='DP Result', figsize=(1000,50))
    for i in range(len(actionVec)):
        if actionVec[i] == 1:
            plt.scatter(i, priceVec[i], s=10, c='green')
        if actionVec[i] == -1:
            plt.scatter(i, priceVec[i], s=10, c='red')
    plt.show()
    fig.figure.savefig('test.png', dpi=500)
    print('find')

if __name__ == '__main__':
    SEARCH = False
    result = readCSV()
    priceVec = createSIMUResult(result)['close']
    date = createSIMUResult(result)['date']
    print('Optimizing over %i numbers of transactions.' % (len(priceVec)))
    transFeeRate = float(0.02)
    actionVec = find_optimal_action(priceVec, transFeeRate)
    # plotBUYSELL(date,priceVec,actionVec)
    returnRate = profit_estimate(priceVec, transFeeRate, actionVec)
    buyTime, sellTime = transBS(result, date, actionVec)
    # writeBuySellCSV(result, buyTime, sellTime)
    wealth, Standard_buyTime, Standard_sellTime, earn, wealthArray, fee = calWealth(buyTime, sellTime)
    write2cvs(result, buyTime, sellTime, earn, wealthArray, fee)
    print(wealth)
    print('Return rate: ', returnRate)
