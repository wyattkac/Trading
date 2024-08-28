"""Code to plot histoic data using matplotlib
"""

import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import yfinance as yf

style.use('ggplot')

df = pd.read_csv('msft.csv', parse_dates=True, index_col=0)
df['100ma'] = df['Close'].rolling(window=100).mean() # Creates a 100 day moving average, often used to see up/downtrend in price
df.dropna(inplace=True) # inplace=True is the same as saying df=COMMAND, could've also set min_periods=0
print(df.tail())

ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)

ax1.plot(df.index, df['Close'])
ax1.plot(df.index, df['100ma'])
ax2.plot(df.index, df['Volume'])

plt.show()
