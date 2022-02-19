from cgitb import reset
from collections import defaultdict
import datetime
from email.policy import default
import itertools
import math
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
from statsmodels.tsa.stattools import adfuller

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


def diff_OP(result):
    # first_day = result['9/11/16']
    # for k,v in result.items():
    #   result[k] = v / first_day
    logArr = []
    date = []
    for k, v in result.items():
        result[k] = math.log(v, 10)
        date.append(k)
        logArr.append(result[k])
    diff = [0]
    pre = logArr[0]
    for i in range(1, len(logArr)):
        diff.append(logArr[i] - pre)
        pre = logArr[i]
    df3 = pd.DataFrame({'dif': diff},
                   index=date, columns=['dif'])
    df3.plot(title='diff')
    plt.show()
    return (date, diff)

def write2csv(date, diff):
    dataframe = pd.DataFrame(
        {'date': date, 'diff': diff})
    dataframe.to_csv("diff-%s.csv"%(time.strftime("%m-%d-%H-%M", time.localtime())) , index=False, sep=',')

def checkADF(diff):
    # for i in range(60, 400):
    #     res = [0] * (len(diff) // i + 1)
    #     # 将 diff 分为 i 个数组
    #     cuts = np.array_split(diff, i)
    #     for cut in cuts:
    #         adf_res = adfuller(cut)
    #         print(adf_res)
    adf_res = adfuller(diff)
    print(adf_res)



if __name__ == '__main__':
    result = readCSV()
    date, diff = diff_OP(result)
    write2csv(date, diff)
    # checkADF(diff)
