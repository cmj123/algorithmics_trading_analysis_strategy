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
f = yf.download(["GOOG"]) # ,"APPL"
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
f["signal_long"] = np.nan
f["yesterday_rsi"] = f["rsi"].shift(1)

# Define the Open long signal (RSI yesterday < 55 and RSI today > 55)
f.loc[(f["rsi"]>neutral_buy)&(f["yesterday_rsi"]<neutral_buy), "signal_long"] =  1

# Close Long Signal (RSI yesterday > 55) and (RSI today < 55) False signal
f.loc[(f["rsi"] < neutral_buy)& (f["yesterday_rsi"] > neutral_buy), "signal_long" ] =  0

# Close long Signal (RSI yesterday > 70) and (RSI today < 70) Over buy signal
f.loc[(f["rsi"] < overbuy)& (f["yesterday_rsi"] > overbuy), "signal_long" ] =  0

## Visualisation
# Select all signal in a index to plot this point
idx_open = f.loc[f["signal_long"]==1].loc["2010"].index 
idx_close =  f.loc[f["signal_long"]==0].loc["2010"].index 

# Adapt the size of the graph
plt.figure(figsize=(15,8))

#Plot the points of the open long signal in green
plt.scatter(f.loc[idx_open]["rsi"].index, f.loc[idx_open]["rsi"].loc["2010"], color = "#57CE95", marker="^" )

# Plot the point of the close long signal in blue 
plt.scatter(f.loc[idx_close]["rsi"].index, f.loc[idx_close]["rsi"].loc["2010"], color="#669fee", marker="o")
# plt.scatter(f[["signal_long"]].loc["2021"].index, f[["signal_long"]].loc["2021"])

# Plot the rsi to be sure that the conditions are completed
plt.plot(f["rsi"].loc["2010"].index, f["rsi"].loc["2010"], alpha=0.35)
plt.show()