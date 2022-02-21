from collections import defaultdict
import datetime
import itertools
import time
from calculateWealth import calWealth_MAIN
from sympy import total_degree
# import matplotlib.pyplot as plt
import talib as ta
import pandas as pd
import csv
import numpy as np
from datetime import datetime
from AUTO_ARIMA import AUTO_ARIMA_MAIN
from DP import DP_MAIN
from Linear import Linear_MAIN

from MACD import MACD_MAIN


bit_data = pd.read_csv('./BCHAIN-MKPRU.csv')
bit_data.index = bit_data.Date
# 从 30 天开始预测
# for i in range(1000, len(bit_data)):
bit_df_everyday = bit_data.head(1000)
MACD_action = MACD_MAIN(bit_df_everyday)
DP_action = DP_MAIN(bit_df_everyday)
LINEAR_action = Linear_MAIN(bit_df_everyday)
AUTO_ARIMA_action = AUTO_ARIMA_MAIN(bit_df_everyday)
total_action = pd.DataFrame({'MACD': MACD_action,'AARIMA': AUTO_ARIMA_action, 'LINEAR': LINEAR_action, 'DP': DP_action},index=bit_df_everyday['Date'])
total_action.to_csv("FINAL.csv", index=True, sep=',')
print('fuck MCM')