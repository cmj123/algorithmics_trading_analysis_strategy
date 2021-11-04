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
print(f.tail())
# Plot the RSI with the overbuy and oversell threshold
plt.figure(figsize=(15, 8))

# View the RSI 
plt.plot(f["rsi"].loc["2021"])

# View horizontal line for the overbuy threshold (RSI = 70)
plt.axhline(70, color="#57CE95")

# View horizontal line for the oversell thresold (RSI = 30)
plt.axhline(30, color="#CE5757")

# Put a title 
plt.title("RSI with thresholds")

# Put a legend
plt.legend(["RSI", "Overbuy", "Oversell"])

plt.show()