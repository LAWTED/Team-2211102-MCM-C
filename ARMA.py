# ARMA预测对数收益率
from cgitb import reset
import csv
import datetime
import itertools
import re
from matplotlib.pyplot import close
# from unittest import result
import pandas as pd
import matplotlib.pylab as plt
# import seaborn as sns
from matplotlib.pylab import style
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
import seaborn as sns

style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# stockFile = 'data/logReturn.csv'
# stock = pd.read_csv(stockFile, index_col=0, parse_dates=[0])#将索引index设置为时间，parse_dates对日期格式处理为标准格式。
# print(stock.head(10))
def readCSV():
    # 读取csv至字典
    csvFile = open("./diff-02-19-22-17.csv", "r")
    reader = csv.reader(csvFile)

    # 建立空字典
    result = {}
    for item in reader:
        # 忽略第一行
        if reader.line_num == 1:
            continue
        result[item[0]] = float(item[1])

    csvFile.close()
    return result


def createTRUEResult(result):
    df = []
    # 生成每日历史数据
    for i in range(1,len(result)):
        df.append(dict(itertools.islice(result.items(), i)))

    for i in range(len(df)):
        df2 = {
            'date': [],
            'close': []
        }
        for k, v in df[i].items():
                df2['close'].append(float(v))
                df2['date'].append(k)
        if len(df2['close']) == 1000:
            operate(df2['date'][-50:], df2['close'][-50:])

#
def operate(date, close):
    df2 = {
        'date': date,
        'close': close
    }
    # acf = plot_acf(df2['close'], lags=20)
    # plt.title("ACF")
    # acf.show()
    # print(acf)
    # pacf = plot_pacf(df2['close'], lags=20)
    # plt.title("PACF")
    # pacf.show()
    # print(pacf)
    # fit model
    begin_pre = datetime.datetime.strptime(df2['date'][-2], '%m/%d/%y')
    end_pre = datetime.datetime.strptime(df2['date'][-2], '%m/%d/%y')+datetime.timedelta(days=3)
    df2 = pd.DataFrame({'close': df2['close'], 'date': df2['date']},
                       index=df2['date'], columns=['close', 'date'])
    # df2['date'] = pd.to_datetime(df2['date'])
    # df2.set_index("date", inplace=True)
    # p, d, q
    # model = ARIMA(df2['close'], order=(1, 1, 1))
    # AR{P} MA{Q}
    # for p in range(1,11):
    #     for q in range(1,11):
    model = ARIMA(df2['close'], order=(1, 2, 1))
    result = model.fit()
    print(result.summary())#统计出ARIMA模型的指标
    pred = result.predict(begin_pre, end_pre,dynamic=True, typ='levels')#预测，指定起始与终止时间。预测值起始时间必须在原始数据中，终止时间不需要
    print (pred)
    # df2['date'] = pd.to_datetime(df2['date'])
    # df2.set_index("date", inplace=True)
    # model = ARIMA(df2['close'], order=(1, 1, 1))
    # result = model.fit()
    # print(result.summary())#统计出ARIMA模型的指标
    # pred = result.predict('20160911', '8/12/18',dynamic=True, typ='levels')#预测，指定起始与终止时间。预测值起始时间必须在原始数据中，终止时间不需要
    # print (pred)




if __name__ == '__main__':
    result=readCSV()
    createTRUEResult(result)
