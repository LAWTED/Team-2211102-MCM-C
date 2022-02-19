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

