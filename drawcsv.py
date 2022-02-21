from multiprocessing import Value
import pandas as pd
import time


df = pd.read_csv('./FINIAL.csv')
fig = df.plot(title="FINAL")
MACD_sum = 0
LINEAR_sum = 0
AARIMA_sum = 0
M = 0
L = 0
A = 0
D = 0
# actionVec = df['DP']
# soft = [0] * len(actionVec)
# for ind, v in enumerate(actionVec):
#     if v != 0:
#         for t in range(ind-2,ind+3):
#             soft[t] = v
# df['DP'] = soft
def pro(Action):
    p = 0
    after = [0] * len(Action)
    while(p < len(Action)):
        if Action[p] == 1:
            sell = []
            while p < len(Action)-1 and Action[p] == 1:
                p += 1
                sell.append(p)
            after[sum(sell)//len(sell)] = 1
        if Action[p] == -1:
            buy = []
            while p < len(Action)-1 and Action[p] == -1:
                p += 1
                buy.append(p)
            after[sum(buy)//len(buy)] = -1
        else:
            p += 1
    # df = pd.DataFrame({'after': after})
    # df.to_csv("AFTER-%s.csv"%(time.strftime("%m-%d-%H-%M", time.localtime())) , index=False, sep=',')
    return after

df['AARIMA'] = pro(df['AARIMA'])

def calRIGHT(df):
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

fig1 = df['AARIMA'].plot()

fig.figure.savefig('FINAL.png', dpi=500)
# fig1.figure.savefig('AARIMA.png', dpi=500)