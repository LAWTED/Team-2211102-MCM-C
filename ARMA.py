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
    csvFile = open("./BCHAIN-MKPRU.csv", "r")
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
    predict = []
    pre_date = []
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
        # if len(df2['close']) == 500:
        #     operate(df2)
        # if len(df2['close']) == 100:
        #     operate(df2)
        if len(df2['close']) > 30:
            operate(df2['date'][-20:], df2['close'][-20:], predict, pre_date)
    dataframe = pd.DataFrame({'predict': predict},
                   index=pre_date, columns=['predict'])
    dataframe.plot(title='predict')
    plt.show()
    print(predict,pre_date)

#
def operate(date, close, predict, pre_date):
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
    # results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(1,8)], columns=['MA{}'.format(i) for i in range(1,8)])
    # for p in range(1,5):
    #     for q in range(1,5):
    model = sm.tsa.SARIMAX(df2['close'], order=(1, 0, 1))
    result = model.fit()
    #         results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = result.bic

    # results_bic = results_bic[results_bic.columns].astype(float)
    # fig, ax = plt.subplots(figsize=(10, 8))
    # ax = sns.heatmap(results_bic,
    #                 mask=results_bic.isnull(),
    #                 ax=ax,
    #                 annot=True,
    #                 fmt='.2f',
    #                 );
    # ax.set_title('BIC');
    # plt.show()
    # print(result.summary())#统计出ARIMA模型的指标
    pred = result.predict(begin_pre, end_pre,dynamic=True, typ='levels')#预测，指定起始与终止时间。预测值起始时间必须在原始数据中，终止时间不需要
    pre_date.append(pred.keys()[-1])
    predict.append(pred[-1])
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
