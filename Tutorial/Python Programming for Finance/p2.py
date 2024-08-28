"""Code to plot historic data using pandas
"""

import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import yfinance as yf

style.use('ggplot')

df = pd.read_csv('msft.csv', parse_dates=True, index_col=0)
#print(df.head())

df['Close'].plot()
plt.show()
