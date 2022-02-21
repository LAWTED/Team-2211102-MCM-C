import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
# %matplotlib inline

from datetime import datetime
df_original = pd.read_csv('./BCHAIN-MKPRU.csv')
df_original.index = df_original.Date
df_gold = pd.read_csv('./NEW-GOLD.csv')
df_gold.index = df_gold.Date

def linear(data):
    day = len(data)
    data = data[-30:]
    x = np.array([i for i in range(len(data['Value']))])
    y = data['Value']

    # # 用pandas读取csv
    # data = pd.read_csv("../data/price_info.csv")
    # x = data['square_feet']
    # y = data['price']

    # 构造X列表和Y列表，reshape(-1,1)改变数组形状，为只有一个属性
    x = x.reshape(-1, 1)
    y = y.values.reshape(-1, 1)

    # 构造回归对象
    model = LinearRegression()
    model.fit(x, y)
    # 获取预测值
    predict_y = model.predict(x)

    # 构造返回字典
    predictions = {}
    predictions['intercept'] = model.intercept_  # 截距值
    predictions['coefficient'] = model.coef_    # 回归系数（斜率值）
    predictions['predict_value'] = predict_y

    # 绘出图像
    # 绘出已知数据散点图
    plt.scatter(x, y, color='blue')
    # 绘出预测直线
    fig = plt.plot(x, predict_y, color='red', linewidth=4)

    plt.title('predict the house price')
    plt.xlabel('square feet')
    plt.ylabel('price')
    plt.savefig(f'./LINEAR/{day}-FFT.png', dpi=500)
    plt.cla()
    return predict_y
    # plt.show()

def getTrend(predict_y):
    trend = (predict_y[-1] - predict_y[0]) / predict_y[0] * 100
    # 1 buy | -1 sell | 0 hold
    Action = 0
    if trend > 10:
        Action = 1
    elif trend < 0 and trend < -10:
        Action = -1
    else:
        Action = 0
    # 实际高于预估 预估上涨 买
    return (Action)

def plotBUYSELL(data, actionVec):
    date = data['Date'][30:]
    priceVec = data['Value'][30:]
    actionVec[0] = 0
    dataframe2 = pd.DataFrame({'price': priceVec}, index=date)
    fig = dataframe2.plot(title='ARIMA Result', figsize=(500,50))
    for i in range(len(actionVec)):
        if actionVec[i] == 1:
            plt.scatter(date[i], priceVec[i], s=10, c='green')
        if actionVec[i] == -1:
            plt.scatter(date[i], priceVec[i], s=10, c='red')
    fig.figure.savefig('LINEAR_BS.png', dpi=100)
    plt.cla()

if __name__ == '__main__':
    Action = []
    for i in range(30,1800,100):
        py = linear(df_original[:i])
        operate = getTrend(py)
        Action.append(operate)
        # regression_GOLD, df_GOLD, day_GOLD = linear(df_gold[:i])
    plotBUYSELL(df_original, Action)
    print(Action)