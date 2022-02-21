from multiprocessing import Value
import pandas as pd
import time

from datetime import datetime
# df_original = pd.read_csv('./BCHAIN-MKPRU.csv')
# df_original.index = df_original.Date
df_gold = pd.read_csv('./gold-02-21-07-43.csv')
# df_gold = pd.DataFrame({'Date': df_gold['Date'], 'Value': df_gold['Value']}, columns=['Date', 'Value'])
# for k in df_original['Date']:
#   if k not in df_gold['Date']:
#     df_gold.loc[k] = [k,0]
# print(df_gold)

df_gold['Date'] = df_gold['Date'].map(lambda x :datetime.strptime(x, '%Y-%m-%d'))
df_gold['Date'] = df_gold['Date'].map(lambda x :datetime.strftime(x, '%#m/%#d/%y'))


# df_gold = df_gold.sort_values(by = 'Date')

# v = df_gold['Value']
# pre = 0
# for i in range(len(v)):
#   if v[i] == 0:
#     v[i] = pre
#   pre = v[i]
# df_gold['Value'] = v

df_gold.to_csv("gold-%s.csv"%(time.strftime("%m-%d-%H-%M", time.localtime())) , index=False, sep=',')