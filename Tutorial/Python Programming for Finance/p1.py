"""'Code to pull historic data off of Yahoo
"""

import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import yfinance as yf # Have to use yfinance as pandas_datareader doesn't work https://aroussi.com/post/python-yahoo-finance

style.use('ggplot')

start = dt.datetime(2015, 12, 30)
end = dt.datetime.now()

stockName = "msft"
msft = yf.Ticker(stockName)
df = msft.history(start=start, end=end) # auto_adjust defaults to true (Close is actually Adj Close)

print(df.tail()) # Good for troubleshooting

df.to_csv(stockName+'.csv')
