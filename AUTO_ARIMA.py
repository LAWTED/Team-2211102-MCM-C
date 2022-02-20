import csv
from datetime import datetime
from operator import mod
import os
import sys
import math
import warnings
import itertools
import numpy as np
import pandas as pd
# import scrapbook as sb
import matplotlib.pyplot as plt

from pmdarima.arima import auto_arima

pd.options.display.float_format = "{:,.2f}".format
np.set_printoptions(precision=2)
warnings.filterwarnings("ignore")

print("System version: {}".format(sys.version))

# Forecasting settings
N_SPLITS = 1
HORIZON = 5  # Forecast 2 Days
GAP = 2
FIRST_WEEK = 40
LAST_WEEK = 138

# Parameters of ARIMA model
params = {
    "seasonal": False,
    "start_p": 0,
    "start_q": 0,
    "max_p": 5,
    "max_q": 5,
    "m": 52,
}


def readCSV():
    # 读取csv至字典
    csvFile = open("BCHAIN-MKPRU.csv", "r")
    reader = csv.reader(csvFile)
    date = []
    price = []
    # 建立空字典
    result = {}
    for item in reader:
        # 忽略第一行
        if reader.line_num == 1:
            continue
        result[item[0]] = float(item[1])

    csvFile.close()
    for k, v in result.items():
        result[k] = math.log(v)
        date.append(datetime.strptime(k, '%m/%d/%y'))
        price.append(result[k])
    return (result, date, price)


def createDF(date, price):
    period = 30
    date = date[-period:]
    price = price[-period:]
    mid = int(period * 0.7)
    train_df = pd.DataFrame({'date': date[:mid], 'price': price[:mid]},
                            index=date[:mid], columns=['date', 'price'])
    test_df = pd.DataFrame({'date': date[mid:], 'price': price[mid:]},
                           index=date[mid:], columns=['date', 'price'])
    return (train_df, test_df)


def train(train_ts):
    train_ts = np.array(train_ts.logmove)

    model = auto_arima(
        train_ts,
        seasonal=params["seasonal"],
        start_p=params["start_p"],
        start_q=params["start_q"],
        max_p=params["max_p"],
        max_q=params["max_q"],
        stepwise=True,
    )

    model.fit(train_ts)


def MAPE(predictions, actuals):
    """
    Implements Mean Absolute Percent Error (MAPE).

    Args:
        predictions (array like): a vector of predicted values.
        actuals (array like): a vector of actual values.

    Returns:
        numpy.float: MAPE value
    """
    if not (isinstance(actuals, pd.Series) and isinstance(predictions, pd.Series)):
        predictions, actuals = pd.Series(predictions), pd.Series(actuals)

    return ((predictions - actuals).abs() / actuals).mean()


# 传入每日数据
def trainEveryDay(date, price):
    day = len(date)
    print(f'This is SIMU of the {day} DAY')
    train_df, test_df = createDF(date, price)
    train_ts = np.array(train_df.price)

    model = auto_arima(
        train_ts,
        seasonal=params["seasonal"],
        start_p=params["start_p"],
        start_q=params["start_q"],
        max_p=params["max_p"],
        max_q=params["max_q"],
        stepwise=True,
    )

    model.fit(train_ts)
    print(model.summary())
    # model.plot_diagnostics(figsize=(10, 8))
    # plt.show()
    preds = model.predict(n_periods=GAP + HORIZON - 1)
    predictions = np.round(np.exp(preds[-HORIZON:]))
    test_date = test_df.date[:HORIZON]
    pred_df = pd.DataFrame({"price": predictions},
                           index=test_date, columns=['price'])
    test_ts = test_df.head(HORIZON)
    train_ts = train_df
    # ts 为 e次方后数据
    test_ts.price = np.round(np.exp(test_ts.price))
    train_ts.price = np.round(np.exp(train_ts.price))
    all_ts = pd.concat([train_ts, test_ts])
    ax = all_ts.price.plot()
    # plt.show()
    fig = pred_df.price.plot(ax=ax, title=f'AUTOARIMA of {day}')
    fig.figure.savefig(f'./AUTO_ARIMA_PNGS/{day}.png', dpi=500)
    # plt.show()

    # MAPE 验证
    # combined = pd.merge(pred_df, test_ts, on=["date"], how="left")
    # metric_value = MAPE(combined.predictions, combined.price) * 100
    # print(f"MAPE of the forecasts is {metric_value}%")
    # print(combined)
    # print(df_all)


if __name__ == '__main__':
    result, date, price = readCSV()
    trainEveryDay(date[:30], price[:30])
