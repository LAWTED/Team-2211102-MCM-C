import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df_original = pd.read_csv('./BCHAIN-MKPRU.csv')
df_original.index = df_original.Date
# df_original = df_original.sort_values(by='Date', ascending=True)
# print(df_original.head(10))
# df = df_original[:1147]

def operate(df):
    day = len(df)
    df = df[['Value']]
    # print(df.head(10))

    # plt.figure(figsize=(15,5))
    # df['Value'].plot(grid=True)
    # plt.ylabel('Price [$]')
    # plt.title('Bitcoin Price')
    # plt.show()

    # 算一阶差分 delta 第一个地方补0
    df['delta'] = np.append(np.array([0]),
                            np.diff(df['Value'].values))
    # print(df['delta'].head(10))

    # 算fft结果 sp
    sp = np.fft.fft(df['delta'].values)
    # print(sp[:10])

    # 算出θ角度值，幅度值和频率
    df['theta'] = np.arctan(sp.imag/sp.real)
    numValues = len(df)
    numValuesHalf = numValues / 2
    df['amplitude'] = np.sqrt(sp.real**2 + sp.imag**2)/numValuesHalf
    df['freq'] = np.fft.fftfreq(sp.size, d=1)
    # print(df.head(5))

    # 画出对应的点阵图
    # plt.figure(figsize=(15,5))
    # plt.plot(df['freq'],df['amplitude'].values, '.')
    # plt.axvline(x=0, ymin=0, ymax = 1, linewidth=1, color='r')
    # plt.ylabel('Amplitude [$]', fontsize=12)
    # plt.xlabel('Frequency [days]', fontsize=12)
    # plt.title('Frequency Domain', fontsize=18)
    # plt.grid()
    # plt.show()

    # 只取在范围之外的点阵
    # take the left half of the mirror image (positive frequency values)
    # as well as filter out any frequency with an amplitude of over 3 standard deviations.
    # 幅度均值meanamp 浮动标准差stdAmp
    # meanAmp = df['amplitude'].mean()
    # stdAmp = df['amplitude'].std()
    # # 幅度大于 均值加上三倍标准差
    # dominantAmpCheck = df['amplitude'] > (3*stdAmp + meanAmp)
    # positiveFreqCheck = df['freq'] > 0
    # dominantAmp = df[dominantAmpCheck & positiveFreqCheck]['amplitude']
    # dominantFreq = df[dominantAmpCheck & positiveFreqCheck]['freq']
    # dominantTheta = df[dominantAmpCheck & positiveFreqCheck]['theta']

    # plt.figure(figsize=(15,5))
    # plt.plot(dominantFreq, dominantAmp, 'o')
    # plt.ylabel('Amplitude [$]', fontsize=12)
    # plt.xlabel('Frequency [days]', fontsize=12)
    # plt.title('Frequency Domain \n(Dominant & Positive)', fontsize=18)
    # plt.grid()
    # # plt.show()

    # regressionDelta = 0
    # for n in range(len(dominantTheta)):
    #     shift = dominantTheta[n]
    #     regressionDelta += dominantAmp[n] * np.cos(n * np.array(range(len(df))) + shift)
    # #Converting Delta Time to Time at start value of real data
    # startValue = df['Value'][0]
    # regression = startValue + np.cumsum(regressionDelta)

    # plt.figure(figsize=(15,5))
    # df['Value'].plot(grid=True)
    # plt.plot(regression)
    # plt.ylabel('Stock Price [$]')
    # plt.legend(['Real','Predicted'])

    # rmse = np.sqrt(np.mean((df['Value'].values - regression)**2))

    # plt.title('RMSE = ' + str(rmse), fontsize=15)
    # # plt.show()

    def std_filter(std_value):

        #Getting dominant values based on std_value
        meanAmp = df['amplitude'].mean()
        stdAmp = df['amplitude'].std()
        dominantAmpCheck = df['amplitude'] > (std_value*stdAmp + meanAmp)
        positiveFreqCheck = df['freq'] > 0
        dominantAmp = df[dominantAmpCheck & positiveFreqCheck]['amplitude']
        dominantFreq = df[dominantAmpCheck & positiveFreqCheck]['freq']
        dominantTheta = df[dominantAmpCheck & positiveFreqCheck]['theta']

        #Calculating Regression Delta
        regressionDelta = 0
        for n in range(len(dominantTheta)):
            shift = dominantTheta[n]
            regressionDelta += dominantAmp[n] * np.cos(n * np.array(range(len(df))) + shift)

        #Converting Delta Time to Time at start value of real data
        startValue = df['Value'][0]
        regression = startValue + np.cumsum(regressionDelta)

        #Calculating RMSE
        rmse = np.sqrt(np.mean((df['Value'].values - regression)**2))

        if np.isnan(rmse):
            rmse = 10000000000000

        return rmse

    std_values = []
    rmse_values = []

    for i in np.linspace(0,2,20):
        std_values.append(i)
        rmse_values.append(std_filter(i))

    idx = np.array(rmse_values).argmin()
    minSTD = std_values[idx]
    minRMSE = rmse_values[idx]

    plt.figure(figsize=(15,5))
    plt.plot(std_values, rmse_values)
    plt.plot(minSTD, minRMSE, 'ro')
    plt.ylabel('RMSE')
    plt.xlabel('STD VALUES')
    plt.title('Lowest RMSE = '+str(minRMSE)+'\nSTD Value = '+str(minSTD))
    plt.grid()
    # plt.show()

    #Getting dominant values based on std_value
    meanAmp = df['amplitude'].mean()
    stdAmp = df['amplitude'].std()
    dominantAmpCheck = df['amplitude'] > (minSTD*stdAmp + meanAmp)
    positiveFreqCheck = df['freq'] > 0
    dominantAmp = df[dominantAmpCheck & positiveFreqCheck]['amplitude']
    dominantFreq = df[dominantAmpCheck & positiveFreqCheck]['freq']
    dominantTheta = df[dominantAmpCheck & positiveFreqCheck]['theta']

    #Calculating Regression Delta
    regressionDelta = 0
    for n in range(len(dominantTheta)):
        temp = dominantAmp[n]
        shift = dominantTheta[n]
        regressionDelta += dominantAmp[n] * np.cos(n * np.array(range(len(df))) + shift)

    #Converting Delta Time to Time at start value of real data
    startValue = df['Value'][0]
    regression = startValue + np.cumsum(regressionDelta)

    plt.figure(figsize=(15,5))
    fig = df['Value'].plot(grid=True)
    plt.plot(regression)
    plt.ylabel('Stock Price [$]')
    plt.legend(['Real','Predicted'])

    rmse = np.sqrt(np.mean((df['Value'].values - regression)**2))

    plt.title('RMSE = ' + str(rmse), fontsize=15)
    # plt.show()
    fig.figure.savefig(f'FFT{day}.png', dpi=500)
    print(df)

# for i in range():
operate(df_original[:1147])
    #Calculating Regression Delta
    # regressionDelta = 0
    # for n in range(len(dominantTheta)):
    #     shift = dominantTheta[n]
    #     regressionDelta += dominantAmp[n] * np.cos(n * np.array(range(len(df_original))) + shift)

    # #Converting Delta Time to Time at start value of real data
    # startValue = df['Value'][0]
    # regression = startValue + np.cumsum(regressionDelta)

    # plt.figure(figsize=(15,5))
    # df_original['Value'].plot(grid=True)
    # plt.plot(regression)
    # plt.ylabel('Stock Price [$]')
    # plt.legend(['Real','Predicted'])

    # plt.axvline(x=1148, ymin=0, ymax = 1, linewidth=2, color='r')

    # rmse = np.sqrt(np.mean((df_original['Value'].values - regression)**2))

    # plt.title('RMSE = ' + str(rmse), fontsize=15)
    # plt.show()