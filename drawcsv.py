from datetime import datetime
from multiprocessing import Value
from matplotlib import pyplot as plt
import pandas as pd
import time
from MACD import write2cvs

from calculateWealth import calWealth_MAIN

bit_data = pd.read_csv('./BCHAIN-MKPRU.csv')
bit_data.index = bit_data.Date
# 从 30 天开始预测
# for i in range(1000, len(bit_data)):


def drawcsv1(bit_df_everyday,df):
    def plotBUYSELL(date, priceVec, actionVec, type):
        if actionVec[0] == -1:
            actionVec[0] = 0
        dataframe2 = pd.DataFrame({'date': date, 'price': priceVec})
        fig = dataframe2.plot(
            title=f'{type}-BUY-AND-SELL-TIME', figsize=(10, 3))
        for i in range(len(actionVec)):
            if actionVec[i] == 1:
                plt.scatter(i, priceVec[i], s=100, c='green')
            if actionVec[i] == -1:
                plt.scatter(i, priceVec[i], s=100, c='red')
        fig.figure.savefig(f'./LAST/{type}-BUY-AND-SELL-TIME.png', dpi=100)

    def calRIGHT(df):
        MACD_sum = 0
        LINEAR_sum = 0
        AARIMA_sum = 0
        M = 0
        L = 0
        A = 0
        D = 0
        for index, row in df.iterrows():
            if row['MACD'] != 0:
                if row['MACD'] == row['DP']:
                    MACD_sum += 1
                M += 1
            if row['LINEAR'] != 0:
                if row['LINEAR'] == row['DP']:
                    LINEAR_sum += 1
                L += 1
            if row['AARIMA'] != 0:
                if row['AARIMA'] == row['DP']:
                    AARIMA_sum += 1
                A += 1
            if row['DP'] != 0:
                D += 1
        print(MACD_sum/M * 100, LINEAR_sum/L * 100,
              AARIMA_sum/A * 100, [M, L, A, D])

    def BS2action(buyTime, sellTime, date):
        action = [0] * len(date)
        for i, v in enumerate(date):
            if datetime.strptime(v, '%m/%d/%y') in buyTime:
                action[i] = 1
            elif datetime.strptime(v, '%m/%d/%y') in sellTime:
                action[i] = -1
        return action

    def write2cvs(wealth, buyTime, sellTime, earn, wealthArray, type):
        dataframe = pd.DataFrame(
            {'buy_time': buyTime, 'sell_time': sellTime, 'earn': earn, 'wealth': wealthArray})
        dataframe.to_csv("./LAST/%s-%s.csv" % (type, time.strftime("%m-%d-%H-%M",
                         time.localtime())), index=False, sep=',')
        print([type, wealth])
        return (type, wealth)

    type_Array = []
    wealth_Array = []
    for type in ('MACD', 'AARIMA', 'LINEAR', 'DP'):
        wealth, buyTime, sellTime, earn, wealthArray = calWealth_MAIN(
            bit_df_everyday['Value'], bit_df_everyday['Date'], df[type])
        type, wealth = write2cvs(wealth, buyTime, sellTime, earn, wealthArray, type)
        type_Array.append(type)
        wealth_Array.append(wealth)
        filter_action = BS2action(buyTime, sellTime, bit_df_everyday['Date'])
        plotBUYSELL(bit_df_everyday['Date'],
                    bit_df_everyday['Value'], filter_action, type)
    FCK = pd.DataFrame({'type': type_Array, 'wealth': wealth_Array})
    FCK.to_csv("./LAST/FUCK-%s.csv" % ( time.strftime("%m-%d-%H-%M",
                         time.localtime())), index=False, sep=',')
    # wealth, buyTime, sellTime, earn, wealthArray = calWealth_MAIN(bit_df_everyday['Value'],bit_df_everyday['Date'], df['AARIMA'])
    # write2cvs(wealth, buyTime, sellTime, earn, wealthArray,'AARIMA')
    # wealth, buyTime, sellTime, earn, wealthArray = calWealth_MAIN(bit_df_everyday['Value'],bit_df_everyday['Date'], df['LINEAR'])
    # write2cvs(wealth, buyTime, sellTime, earn, wealthArray,'LINEAR')
    # wealth, buyTime, sellTime, earn, wealthArray = calWealth_MAIN(bit_df_everyday['Value'],bit_df_everyday['Date'], df['DP'])
    # write2cvs(wealth, buyTime, sellTime, earn, wealthArray,'DP')
    # hold = 1000/bit_df_everyday['Value'][0]
    # print(bit_df_everyday['Value'][-1], bit_df_everyday['Value'][0])
    # print(['stay', (hold * bit_df_everyday['Value'][-1] - (bit_df_everyday['Value'][0] + bit_df_everyday['Value'][-1]) * 0.02)])

    # # fig1 = df['AARIMA'].plot()
    # calRIGHT(df)
    # # fig.figure.savefig('FINAL.png', dpi=500)
    # # fig1.figure.savefig('AARIMA.png', dpi=500)
