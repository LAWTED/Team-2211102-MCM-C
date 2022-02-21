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
    # plt.show()

for i in range(40,1800):
    linear(df_original[:i])
    # regression_GOLD, df_GOLD, day_GOLD = linear(df_gold[:i])