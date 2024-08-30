"""Code to make a candlestick/OHLC graph from histoic data using mplfinance and pandas
"""

import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
from mplfinance.original_flavor import candlestick_ohlc # now can use mplfinance.plot instead
import matplotlib.dates as mdates
import pandas as pd
import yfinance as yf

style.use('ggplot')

df = pd.read_csv('msft.csv', parse_dates=True, index_col=0)
df.index = pd.to_datetime(df.index, utc=True)

df_ohlc = df['Close'].resample('10D').ohlc()
df_volume = df['Volume'].resample('10D').sum()

df_ohlc.reset_index(inplace=True)
df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)

ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)
ax1.xaxis_date()

candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')
ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
plt.show()
