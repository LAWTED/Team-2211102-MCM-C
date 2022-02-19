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

# 王瑶:
# 1.标准化 除以第一天的值

# 王瑶:
# 2. 取对数 以十为底

# 王瑶:
# 3. 一阶差分


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
    # df3.plot(title='diff')
    # plt.show()
    return (date, diff)

def write2csv(date, diff):
    dataframe = pd.DataFrame(
        {'date': date, 'diff': diff})
    dataframe.to_csv("diff-%s.csv"%(time.strftime("%m-%d-%H-%M", time.localtime())) , index=False, sep=',')

if __name__ == '__main__':
    result = readCSV()
    date, diff = diff_OP(result)
    write2csv(date, diff)


