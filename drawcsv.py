from multiprocessing import Value
import pandas as pd
import time
from MACD import write2cvs

from calculateWealth import calWealth_MAIN

bit_data = pd.read_csv('./BCHAIN-MKPRU.csv')
bit_data.index = bit_data.Date
# 从 30 天开始预测
# for i in range(1000, len(bit_data)):
bit_df_everyday = bit_data.head(1000)
df = pd.read_csv('./FINIAL.csv')
fig = df.plot(title="FINAL")

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
  print(MACD_sum/M * 100,LINEAR_sum/L * 100,AARIMA_sum/A * 100,[M,L,A,D])

def write2cvs(wealth, buyTime, sellTime, earn, wealthArray, type):
    dataframe = pd.DataFrame(
        {'buy_time': buyTime, 'sell_time': sellTime, 'earn': earn, 'wealth': wealthArray})
    dataframe.to_csv("%s-%s.csv"%(type, time.strftime("%m-%d-%H-%M", time.localtime())) , index=False, sep=',')
    print([type, wealth])

wealth, buyTime, sellTime, earn, wealthArray = calWealth_MAIN(bit_df_everyday['Value'],bit_df_everyday['Date'], df['MACD'])
write2cvs(wealth, buyTime, sellTime, earn, wealthArray,'MACD')
wealth, buyTime, sellTime, earn, wealthArray = calWealth_MAIN(bit_df_everyday['Value'],bit_df_everyday['Date'], df['AARIMA'])
write2cvs(wealth, buyTime, sellTime, earn, wealthArray,'AARIMA')
wealth, buyTime, sellTime, earn, wealthArray = calWealth_MAIN(bit_df_everyday['Value'],bit_df_everyday['Date'], df['LINEAR'])
write2cvs(wealth, buyTime, sellTime, earn, wealthArray,'LINEAR')
wealth, buyTime, sellTime, earn, wealthArray = calWealth_MAIN(bit_df_everyday['Value'],bit_df_everyday['Date'], df['DP'])
write2cvs(wealth, buyTime, sellTime, earn, wealthArray,'DP')
hold = 1000/bit_df_everyday['Value'][0]
print(bit_df_everyday['Value'][-1], bit_df_everyday['Value'][0])
print(['stay', (hold * bit_df_everyday['Value'][-1] - (bit_df_everyday['Value'][0] + bit_df_everyday['Value'][-1]) * 0.02)])


# fig1 = df['AARIMA'].plot()
calRIGHT(df)
# fig.figure.savefig('FINAL.png', dpi=500)
# fig1.figure.savefig('AARIMA.png', dpi=500)