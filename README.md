# Team-2211102-MCM-C

1. 黄金交易日期研究
- 周末不交易
- （不重要但留着了）全球各大金市的交易时间，以伦敦时间为准，形成伦敦、纽约 （芝加哥）连续不停的黄金交易：伦敦每天上午10：30的早盘定价揭开北美金市的序幕。纽约、芝加哥等先后开盘，当伦敦下午定价后，纽约等地仍在交易，此时香港亦开始进行交易。伦敦的尾市影响美国的早市价格，美国的尾市会影响香港的开市价格，而香港的尾市和美国的收盘价又会影响伦敦的开市价，如此循环。正常交易时间为北京时间周一（08：00am）至周六（夏令时凌晨01：30am，冬令时令凌晨02：30am），节假日及国际市场休市则停止交易。
2. 牛熊市
- 牛市就是上行行情或者震荡上行行情，高点破高，低点也破高，如果低点不与上一个高点重合就是上行行情，如果低点与上一个高点有重合，就是震荡上行；熊市同理正好相反
- 说人话就是：牛市是持续上涨，熊市是持续下跌
- bull market and bear market
3. 乖离率
- 乖离率就是市场收盘价与移动平均线之间的差距百分比
4. MACD
- Moving average convergence divergence (MACD) is a trend-following momentum indicator that shows the relationship between two moving averages of a security’s price.The MACD is calculated by subtracting the 26-period exponential moving average (EMA) from the 12-period EMA.
  ### MACD Formula:
  MACD=12-Period EMA − 26-Period EMA
  - A nine-day EMA of the MACD called the "signal line," is then plotted on top of the MACD line, which can function as a trigger for buy and sell signals
  - Moving average convergence divergence (MACD) indicators can be interpreted in several ways, but the more common methods are crossovers, divergences, and rapid rises/falls.
  ### key takeaways:
  - Moving average convergence divergence (MACD) is calculated by subtracting the 26-period exponential moving average (EMA) from the 12-period EMA.
  - MACD triggers technical signals when it crosses above (to buy) or below (to sell) its signal line.
  - The speed of crossovers is also taken as a signal of a market is overbought or oversold.
  - MACD helps investors understand whether the bullish or bearish movement in the price is strengthening or weakening.
 5. log-return
 - The logarithmic return or continuously compounded return, also known as force of interest, is:

{\displaystyle R_{\mathrm {log} }=\ln \left({\frac {V_{f}}{V_{i}}}\right)}{\displaystyle R_{\mathrm {log} }=\ln \left({\frac {V_{f}}{V_{i}}}\right)}
and the logarithmic rate of return is:

{\displaystyle r_{\mathrm {log} }={\frac {\ln \left({\frac {V_{f}}{V_{i}}}\right)}{t}}}r_{\mathrm{log}} = \frac{\ln\left(\frac{V_f}{V_i}\right)}{t}
or equivalently it is the solution {\displaystyle r}r to the equation:

{\displaystyle V_{f}=V_{i}e^{r_{\mathrm {log} }t}}{\displaystyle V_{f}=V_{i}e^{r_{\mathrm {log} }t}}
where:

{\displaystyle r_{\mathrm {log} }}r_{\mathrm{log}} = logarithmic rate of return
{\displaystyle t}t = length of time period

  >>https://www.investopedia.com/terms/m/macd.asp
5. 快踩一手机器学习！
  In order to learn the features of the modeled task and be able to predict, an LSTM needs to be trained. This process consists in computing the weights and biases of the LSTM by minimizing an objective function, typically RMSE, through some optimization algorithms. Once the model it’s trained on an initial training dataset and validated on a validation set, it is then tested on a real out of sample testing. This ensures that the model did in fact learn useful features and it is not overfitted on the training set, with poor prediction capabilities on new data. The next section analyses the performance of an LSTM applied to the S&P 500.
  重点：While it is true that new machine learning algorithms, in particular deep learning, have been quite successful in different areas, they are not able to predict the US equity market. LSTM just use a value very close to the previous day closing price as prediction for the next day value. This is what would be expected by a model that has no predictive ability.
  >>https://www.blueskycapitalmanagement.com/machine-learning-in-finance-why-you-should-not-use-lstms-to-predict-the-stock-market/
6. 过拟合,数据缺失对深度学习的影响
   - overfitting:It is a common pitfall in deep learning algorithms in which a model tries to fit the training data entirely and ends up memorizing the data patterns and the noise and random fluctuations. These models fail to generalize and perform well in the case of unseen data scenarios, defeating the model's purpose. If the model trains for too long on the training data or is too complex, it learns the noise or irrelevant information within the dataset.
     >>https://www.v7labs.com/blog/overfitting#what-is-overfitting
   - Missing data present various problems. First, the absence of data reduces statistical power, which refers to the probability that the test will reject the null hypothesis when it is false. Second, the lost data can cause bias in the estimation of parameters. Third, it can reduce the representativeness of the samples. Fourth, it may complicate the analysis of the study. Each of these distortions may threaten the validity of the trials and can lead to invalid conclusions.
   >>Kang H. The prevention and handling of the missing data. Korean J Anesthesiol. 2013;64(5):402-406. doi:10.4097/kjae.2013.64.5.402
7. bull market && bear market:
   - bull market, in securities and commodities trading, a rising market. A bull is an investor who expects prices to rise and, on this assumption, purchases a security or commodity in hopes of reselling it later for a profit. A bullish market is one in which prices are generally expected to rise.
   >>https://www.britannica.com/topic/bull-market
   - bear market, in securities and commodities trading, a declining market. A bear is an investor who expects prices to decline and, on this assumption, sells a borrowed security or commodity in the hope of buying it back later at a lower price, a speculative transaction called selling short. The term bear may derive from the proverb about “selling the bearskin before one has caught the bear” or perhaps from selling when one is “bare” of stock.
   >>https://www.britannica.com/topic/bear-market
8. investment cycle:
   cycle covers the period, usually spanning several business cycles, from the time of the Investment until the point where it stops generating cash flows. It includes Capital expenditures, disposals of Fixed assets, and changes in long-term Investments
9. 股票预测的模型，主要是神经网络与传统统计模型、随机模型的讨论：
 - for short term prediction using the time series data, the ARIMA model and the stochastic model can be used interchangeably.For the ANN models, further studies, hybridization of existing models, and adding more independentvariables can improve the neural network models in predicting stock prices. One model can workbetter than other models with particular time series data.

10. 动态规划
https://en.wikipedia.org/wiki/Dynamic_programming
https://www.newgenesiscap.com/deal_427470

11. mixed integer programming model
https://www.solver.com/integer-constraint-programming

# Fintech: Best Time to Buy and Sell Stock with Transaction Fee
* Find the best time to buy and sell stock with transaction fee using Dynamic Programming, implementation in Python.

## Algorithm
- Use **Dynamic Programming** to compute to optimal action sequence along a give price vector.
- DP records the following at each time t:
	- optimal value of money, and
	- optimal value of stock, and
	- the previous action that lead to this optimal value
- **DP initialization:**
	- money = original capital, and
	- stock = buy stock with original capital
- **DP recursion:**
	- calculate optimal money:
		* money = hold yesterday's money, or
		* money = sell all your stocks today
	- calculate optimal stock:
		* stock = hold yesterday's stock, or
		* stock = buy stock today with all your money
- **DP trace back:**
	- You must sell on the last day to maximize profit
	- trace the previous action that lead to this optimal value
- **Action smoothing:**
	- if previous action is the same as the current action, then the action choosen is "hold".

## Result
- Testing on the [SPY dataset](https://finance.yahoo.com/quote/SPY/history?period1=1167580800&period2=1508947200&interval=1d&filter=history&frequency=1d), **return rate is: 212.88675299365727**
