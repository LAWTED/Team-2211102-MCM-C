import datetime
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
df2 = {
  'date': [],
  'close': []
}
for k,v in result.items():
  df2['close'].append(float(v))
  df2['date'].append(k)
buy = []
sell = []

def MACD(df2):
    dif, dea, hist = ta.MACD( np.array(df2['close']), fastperiod=6, slowperiod=30, signalperiod=6)
    df3 = pd.DataFrame({'dif': dif[33:], 'dea': dea[33:], 'hist': hist[33:]},
                       index=df2['date'][33:], columns=['dif', 'dea', 'hist'])
    df3.plot(title='MACD')
    # plt.show()
    # 寻找MACD金叉和死叉
    datenumber = int(df3.shape[0])
    for i in range(datenumber-1):
        if ((df3.iloc[i, 0] <= df3.iloc[i, 1]) & (df3.iloc[i+1, 0] >= df3.iloc[i+1, 1])):
            print("MACD金叉的日期："+df3.index[i+1])
            buy.append(df3.index[i+1])
        if ((df3.iloc[i, 0] >= df3.iloc[i, 1]) & (df3.iloc[i+1, 0] <= df3.iloc[i+1, 1])):
            print("MACD死叉的日期："+df3.index[i+1])
            sell.append(df3.index[i+1])
    return(dif, dea, hist)

# MACD(result)

# print(df2['close'][:200])
# print(df2['date'][:200])
MACD(df2)


buyTime = []
sellTime = []
earn = []
wealthArray = []
def calWealth(buy,sell):
  wealth = 1000
  hold = 0
  # print(len(buy))
  # print(len(sell))
  firstBuyDate = datetime.strptime(buy[0], '%m/%d/%y')
  firstSellDate = datetime.strptime(sell[0], '%m/%d/%y')
  print(buy,sell)
  if firstBuyDate < firstSellDate:
    for i in range(len(buy)):
      hold = wealth / result[buy[i]]
      if result[buy[i]] < result[sell[i]]:
        wealth += hold * (result[sell[i]] - result[buy[i]])
        buyTime.append(buy[i])
        sellTime.append(sell[i])
        earn.append(hold * (result[sell[i]] - result[buy[i]]))
        wealthArray.append(wealth)
        print('buy at'+buy[i]+'sell at' + sell[i] + 'earn ' + hold * (result[sell[i]] - result[buy[i]]))
  # 第一天就买入了
  else:
    buy = ['9/11/16'] + buy
    # buy 98个日期 sell 98个日期
    for i in range(len(buy)):
      # 当前的钱除价格算 份额
      # 只有涨了才卖
      if result[buy[i]] < result[sell[i]]:
        buyTime.append(datetime.strptime(buy[i], '%m/%d/%y')) # 添加买入时间
        sellTime.append(datetime.strptime(sell[i], '%m/%d/%y')) # 卖出
        hold = wealth / result[buy[i]]
        wealth += hold * (result[sell[i]] - result[buy[i]]) # wealth 增加了 卖出价格 减去 买入价格 乘份额
        earn.append(hold * (result[sell[i]] - result[buy[i]]))
        wealthArray.append(wealth)
        print('buy at '+buy[i]+' sell at ' + sell[i] + ' earn ' + str(hold * (result[sell[i]] - result[buy[i]])) + ' NOW Wealth ' + str(wealth))

  return wealth
# total = calWealth(buy,sell)
# print(total)

def write2cvs (buyTime, sellTime, earn, wealthArray):
  dataframe = pd.DataFrame({'buy_time': buyTime, 'sell_time': sellTime , 'earn': earn, 'wealth': wealthArray })
  dataframe.to_csv("operate.csv",index=False,sep=',')

# write2cvs(buyTime, sellTime, earn, wealthArray)