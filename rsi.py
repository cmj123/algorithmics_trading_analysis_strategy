# Import library
import numpy as np
import pandas as pd
from pandas.core.window.rolling import Window
import yfinance as yf 
import matplotlib as mpl 
import matplotlib.pyplot as plt
from matplotlib import cycler
import seaborn as sns
import ta 
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# The code here will allow you to switch your graphics to dark mode for those who choose to code in dark mode
# colors = cycler('color', 
#                 ['#669'])

# import the data
f = yf.download(["PLUG"]) # ,"APPL"
# Compute the rsi 
f["rsi"] = ta.momentum.RSIIndicator(f["Adj Close"], window=14).rsi()

## Creating RSI zone of action
plt.figure(figsize=(15,8))

# View RSI
plt.plot(f["rsi"].loc["2021"])

# View buy zone of action
plt.fill_between(f["rsi"].loc["2021"].index, 55,70, color="#57CE95", alpha=0.5)

# View sell zone of action
plt.fill_between(f["rsi"].loc["2021"].index, 45,30, color="#CE5757", alpha=0.5)

# Put a title 
plt.title("RSI with zone on long buy and short sell")

# Put a legend
plt.legend(["RSI", "Long buy signal", "Short sell zone"])

# plt.show()

## Define RSI - Buying Signals
# define the threshold
# overbuy threshold
overbuy = 70 
# set the lower end of the buy region
neutral_buy = 55

# Put nan values for the signal long columns
f["signal_long"] = 0
f["yesterday_rsi"] = f["rsi"].shift(1)

# Define the Open long signal (RSI yesterday < 55 and RSI today > 55)
f.loc[(f["rsi"]>neutral_buy)&(f["yesterday_rsi"]<neutral_buy), "signal_long"] =  1
print(f[f["signal_long"] == 1])
plt.figure(figsize=(15,8))
f[["signal_long"]].plot()
plt.show()