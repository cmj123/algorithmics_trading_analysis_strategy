# Import library
import numpy as np
from numpy.core.fromnumeric import mean
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

# ## Creating RSI zone of action
# plt.figure(figsize=(15,8))

# # View RSI
# plt.plot(f["rsi"].loc["2021"])

# # View buy zone of action
# plt.fill_between(f["rsi"].loc["2021"].index, 55,70, color="#57CE95", alpha=0.5)

# # View sell zone of action
# plt.fill_between(f["rsi"].loc["2021"].index, 45,30, color="#CE5757", alpha=0.5)

# # Put a title 
# plt.title("RSI with zone on long buy and short sell")

# # Put a legend
# plt.legend(["RSI", "Long buy signal", "Short sell zone"])

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

# # Adapt the size of the graph
# plt.figure(figsize=(15,8))

# #Plot the points of the open long signal in green
# plt.scatter(f.loc[idx_open]["rsi"].index, f.loc[idx_open]["rsi"].loc["2010"], color = "#57CE95", marker="^" )

# # Plot the point of the close long signal in blue 
# plt.scatter(f.loc[idx_close]["rsi"].index, f.loc[idx_close]["rsi"].loc["2010"], color="#669fee", marker="o")
# # plt.scatter(f[["signal_long"]].loc["2021"].index, f[["signal_long"]].loc["2021"])

# # Plot the rsi to be sure that the conditions are completed
# plt.plot(f["rsi"].loc["2010"].index, f["rsi"].loc["2010"], alpha=0.35)
# plt.show()

## RSI - Sell Signal
# Define sell threshold
oversell = 30
neutral_sell = 45

# Put nan values for the signal short columns 
f["signal_short"]=np.nan 


# Define the Open short signal (RSI yesterday>45 and RSI today < 45)
f.loc[(f["rsi"]<neutral_sell) & (f["yesterday_rsi"]>neutral_sell),"signal_short"] = -1

# Define Close short signal (RSI yesterday < 45 and RSI today > 45) False signal
f.loc[(f["rsi"]>neutral_sell) & (f["yesterday_rsi"]<neutral_sell),"signal_short"] = 0

# Define Close short signal (RSI yesterday < 30 and RSI today > 30) 
f.loc[(f["rsi"]>oversell) & (f["yesterday_rsi"]<oversell),"signal_short"] = 0

## Plot sell signal
# Get index positions for close
idx_open = f.loc[f["signal_short"]==-1].loc["2010"].index
idx_close = f.loc[f["signal_short"]==0].loc["2010"].index

# # Adapt the size of the graph
# # plt.figure(figsize=(15,8))

# # Plot the points of the open short signal in red
# plt.scatter(f.loc[idx_open]["rsi"].index, f.loc[idx_open]["rsi"].loc["2010"], color="#CE5757", marker="v")

# # Plot the points of the close short signal in blue
# plt.scatter(f.loc[idx_close]["rsi"].index, f.loc[idx_close]["rsi"].loc["2010"], color="black", marker="o")

# # Plot the rsi to be sure that the conditions are completed
# plt.plot(f["rsi"].loc["2010"].index, f["rsi"].loc["2010"], alpha=0.35)

# # Show the graph
# plt.show()

# Define column - position
f["Position"] = (f["signal_short"].fillna(method="ffill") + f["signal_long"].fillna(method="ffill"))
f.dropna(thresh=10)

# Plot all the signal to be sure 
print(f.tail())

# Plot all the signals
year = "2010"
idx_long = f.loc[f["Position"]==1].loc[year].index
idx_short = f.loc[f["Position"]==-1].loc[year].index

# Adapt the size of the graph
plt.figure(figsize=(15,8))

# Plot the points of the open short signal in red
plt.scatter(f.loc[idx_short]["Adj Close"].index, f.loc[idx_short]["Adj Close"].loc[year], color="#CE5757", marker="v")

# Plot the points of the close short signal in blue
plt.scatter(f.loc[idx_long]["Adj Close"].index, f.loc[idx_long]["Adj Close"].loc[year], color="#57CE95", marker="^")

# # Plot the rsi to be sure that the conditions are completed
plt.plot(f["Adj Close"].loc[year].index, f["Adj Close"].loc[year], alpha=0.35)

# Show the graph
plt.show()

## Compute the perctange change
f["pct"] = f["Adj Close"].pct_change(1)

# Compute the return of the strategy
f["return"] = f["pct"]*f["Position"].shift(0)

f["return"].loc["2010"].cumsum().plot(figsize=(15,8))

plt.show()

## Create a function to do the RSI strategy
def RSI(val, neutral, window):
    """
    Output: The function gives the returns of RSI strategy
    Inputs: -val (type dataframe pandas): Entry values of the stock
            -neutral (float): Value of neutrality i.e. no action zone
            -window (float): rolling period for RSI
    """

    # Print Error if there is no column Adj Close in the dataframe
    if "Adj Close" not in val.columns:
        ValueError("Need to have a columns named Adj Close all computations are about this columns")

    # Calcualte the RSI
    val["rsi"] = ta.momentum.RSIIndicator(val["Adj Close"], window=window).rsi()

    """Long buy Signal"""
    # Set the threshold
    overbuy = 70
    neutral_buy = 50 + neutral 

    # Put nan values for the signal long columns 
    val["signal_long"] = np.nan 
    val["yesterday_rsi"] = val["rsi"].shift(1)

    # Need to define the Open Long Singal (RSI yesterday < 55 and RSI today > 55)
    val.loc[(val["rsi"]>neutral_buy), (val["yesterday_rsi"] < neutral_buy), "signal_long"] = 1

    # Need to define the Close Long Singal (RSI yesterday > 55 and RSI today < 55) False Signal
    val.loc[(val["rsi"]<neutral_buy), (val["yesterday_rsi"] > neutral_buy), "signal_long"] = 0

    # Need to define the Close Long Singal (RSI yesterday > 55 and RSI today < 55) Over buy signal
    val.loc[(val["rsi"]<overbuy), (val["yesterday_rsi"] > overbuy), "signal_long"] = 0

    """Short sell signal"""
    # Set the threshold
    oversell = 30 
    neutral_sell = 50 - neutral

    # Put nan values for the signal short columns
    val["signal_short"] = np.nan 

    # Define the Open  Short signal (RSI yesterday > 45) and (RSI today <  45)
    val.loc[(val["rsi"] < neutral_sell) & (val["rsi_yesterday"] > neutral_sell), "signal_short"] = -1

    # Define the Open  Short signal (RSI yesterday < 45) and (RSI today >  45) False Signal
    val.loc[(val["rsi"] > neutral_sell) & (val["rsi_yesterday"] < neutral_sell), "signal_short"] = 0

    # Define the Open  Short signal (RSI yesterday < 45) and (RSI today >  45) 
    val.loc[(val["rsi"] < oversell) & (val["rsi_yesterday"] > oversell), "signal_short"] = 0