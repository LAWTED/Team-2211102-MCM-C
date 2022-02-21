
from datetime import datetime


def calWealth_MAIN(result, actionVec):
    def transBS(result, actionVec):
        date = result['Date']
        buyTime = []
        sellTime = []
        for ind, v in enumerate(actionVec):
            if v == 1:
                buyTime.append(date[ind])
            if v == -1:
                sellTime.append(date[ind])
        return (buyTime, sellTime)
    buy, sell = transBS(result, actionVec)
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