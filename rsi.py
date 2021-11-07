# Import library
import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
from pandas.core.window.rolling import Window
import yfinance as yf 
from hurst import compute_Hc
import matplotlib as mpl 
import matplotlib.pyplot as plt
from matplotlib import cycler
import seaborn as sns
import ta 
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# The code here will allow you to switch your graphics to dark mode for those who choose to code in dark mode
colors = cycler('color',
                ['#669FEE', '#66EE91', '#9988DD',
                 '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('figure', facecolor='#313233')
plt.rc('axes', facecolor="#313233", edgecolor='none',
       axisbelow=True, grid=True, prop_cycle=colors,
       labelcolor='gray')
plt.rc('grid', color='474A4A', linestyle='solid')
plt.rc('xtick', color='gray')
plt.rc('ytick', direction='out', color='gray')
plt.rc('legend', facecolor="#313233", edgecolor="#313233")
plt.rc("text", color="#C9C9C9")
plt.rc('figure', facecolor='#313233')

# import the data
f = yf.download(["GOOG"]) # ,"APPL"

# # Compute the rsi 
# f["rsi"] = ta.momentum.RSIIndicator(f["Adj Close"], window=14).rsi()

# # ## Creating RSI zone of action
# # plt.figure(figsize=(15,8))

# # # View RSI
# # plt.plot(f["rsi"].loc["2021"])

# # # View buy zone of action
# # plt.fill_between(f["rsi"].loc["2021"].index, 55,70, color="#57CE95", alpha=0.5)

# # # View sell zone of action
# # plt.fill_between(f["rsi"].loc["2021"].index, 45,30, color="#CE5757", alpha=0.5)

# # # Put a title 
# # plt.title("RSI with zone on long buy and short sell")

# # # Put a legend
# # plt.legend(["RSI", "Long buy signal", "Short sell zone"])

# # plt.show()

# ## Define RSI - Buying Signals
# # define the threshold
# # overbuy threshold
# overbuy = 70 
# # set the lower end of the buy region
# neutral_buy = 55

# # Put nan values for the signal long columns
# f["signal_long"] = np.nan
# f["yesterday_rsi"] = f["rsi"].shift(1)

# # Define the Open long signal (RSI yesterday < 55 and RSI today > 55)
# f.loc[(f["rsi"]>neutral_buy)&(f["yesterday_rsi"]<neutral_buy), "signal_long"] =  1

# # Close Long Signal (RSI yesterday > 55) and (RSI today < 55) False signal
# f.loc[(f["rsi"] < neutral_buy)& (f["yesterday_rsi"] > neutral_buy), "signal_long" ] =  0

# # Close long Signal (RSI yesterday > 70) and (RSI today < 70) Over buy signal
# f.loc[(f["rsi"] < overbuy)& (f["yesterday_rsi"] > overbuy), "signal_long" ] =  0

# ## Visualisation
# # Select all signal in a index to plot this point
# idx_open = f.loc[f["signal_long"]==1].loc["2010"].index 
# idx_close =  f.loc[f["signal_long"]==0].loc["2010"].index 

# # # Adapt the size of the graph
# # plt.figure(figsize=(15,8))

# # #Plot the points of the open long signal in green
# # plt.scatter(f.loc[idx_open]["rsi"].index, f.loc[idx_open]["rsi"].loc["2010"], color = "#57CE95", marker="^" )

# # # Plot the point of the close long signal in blue 
# # plt.scatter(f.loc[idx_close]["rsi"].index, f.loc[idx_close]["rsi"].loc["2010"], color="#669fee", marker="o")
# # # plt.scatter(f[["signal_long"]].loc["2021"].index, f[["signal_long"]].loc["2021"])

# # # Plot the rsi to be sure that the conditions are completed
# # plt.plot(f["rsi"].loc["2010"].index, f["rsi"].loc["2010"], alpha=0.35)
# # plt.show()

# ## RSI - Sell Signal
# # Define sell threshold
# oversell = 30
# neutral_sell = 45

# # Put nan values for the signal short columns 
# f["signal_short"]=np.nan 


# # Define the Open short signal (RSI yesterday>45 and RSI today < 45)
# f.loc[(f["rsi"]<neutral_sell) & (f["yesterday_rsi"]>neutral_sell),"signal_short"] = -1

# # Define Close short signal (RSI yesterday < 45 and RSI today > 45) False signal
# f.loc[(f["rsi"]>neutral_sell) & (f["yesterday_rsi"]<neutral_sell),"signal_short"] = 0

# # Define Close short signal (RSI yesterday < 30 and RSI today > 30) 
# f.loc[(f["rsi"]>oversell) & (f["yesterday_rsi"]<oversell),"signal_short"] = 0

# ## Plot sell signal
# # Get index positions for close
# idx_open = f.loc[f["signal_short"]==-1].loc["2010"].index
# idx_close = f.loc[f["signal_short"]==0].loc["2010"].index

# # # Adapt the size of the graph
# # # plt.figure(figsize=(15,8))

# # # Plot the points of the open short signal in red
# # plt.scatter(f.loc[idx_open]["rsi"].index, f.loc[idx_open]["rsi"].loc["2010"], color="#CE5757", marker="v")

# # # Plot the points of the close short signal in blue
# # plt.scatter(f.loc[idx_close]["rsi"].index, f.loc[idx_close]["rsi"].loc["2010"], color="black", marker="o")

# # # Plot the rsi to be sure that the conditions are completed
# # plt.plot(f["rsi"].loc["2010"].index, f["rsi"].loc["2010"], alpha=0.35)

# # # Show the graph
# # plt.show()

# # Define column - position
# f["Position"] = (f["signal_short"].fillna(method="ffill") + f["signal_long"].fillna(method="ffill"))
# f.dropna(thresh=10)

# # Plot all the signal to be sure 
# print(f.tail())

# # Plot all the signals
# year = "2010"
# idx_long = f.loc[f["Position"]==1].loc[year].index
# idx_short = f.loc[f["Position"]==-1].loc[year].index

# # Adapt the size of the graph
# plt.figure(figsize=(15,8))

# # Plot the points of the open short signal in red
# plt.scatter(f.loc[idx_short]["Adj Close"].index, f.loc[idx_short]["Adj Close"].loc[year], color="#CE5757", marker="v")

# # Plot the points of the close short signal in blue
# plt.scatter(f.loc[idx_long]["Adj Close"].index, f.loc[idx_long]["Adj Close"].loc[year], color="#57CE95", marker="^")

# # # Plot the rsi to be sure that the conditions are completed
# plt.plot(f["Adj Close"].loc[year].index, f["Adj Close"].loc[year], alpha=0.35)

# # Show the graph
# plt.show()

# ## Compute the perctange change
# f["pct"] = f["Adj Close"].pct_change(1)

# # Compute the return of the strategy
# f["return"] = f["pct"]*f["Position"].shift(0)

# f["return"].loc["2010"].cumsum().plot(figsize=(15,8))

# plt.show()

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
    val.loc[(val["rsi"]>neutral_buy) & (val["yesterday_rsi"] < neutral_buy), "signal_long"] = 1

    # Need to define the Close Long Singal (RSI yesterday > 55 and RSI today < 55) False Signal
    val.loc[(val["rsi"]<neutral_buy)& (val["yesterday_rsi"] > neutral_buy), "signal_long"] = 0

    # Need to define the Close Long Singal (RSI yesterday > 55 and RSI today < 55) Over buy signal
    val.loc[(val["rsi"]<overbuy)& (val["yesterday_rsi"] > overbuy), "signal_long"] = 0

    """Short sell signal"""
    # Set the threshold
    oversell = 30 
    neutral_sell = 50 - neutral

    # Put nan values for the signal short columns
    val["signal_short"] = np.nan 

    # Define the Open  Short signal (RSI yesterday > 45) and (RSI today <  45)
    val.loc[(val["rsi"] < neutral_sell) & (val["yesterday_rsi"] > neutral_sell), "signal_short"] = -1

    # Define the Open  Short signal (RSI yesterday < 45) and (RSI today >  45) False Signal
    val.loc[(val["rsi"] > neutral_sell) & (val["yesterday_rsi"] < neutral_sell), "signal_short"] = 0

    # Define the Open  Short signal (RSI yesterday < 45) and (RSI today >  45) 
    val.loc[(val["rsi"] < oversell) & (val["yesterday_rsi"] > oversell), "signal_short"] = 0

    """ Compute the returns """
    # Compute the response percentage of variation of the asset
    val["pct"] = val["Adj Close"].pct_change(1)

    # Compute the positions
    val["Position"] = (val["signal_short"].fillna(method="ffill") + val["signal_long"].fillna(method="ffill"))

    # Compute the return of the strategy 
    val["return"] = val["pct"]*val["Position"].shift(1)

    return val["return"]

# Create a beta function
def beta_function(serie):
    # Get SP500 data
    sp500 = yf.download("^GSPC")[["Adj Close"]].pct_change()

    # Change column name
    sp500.columns = ["SP500"]

    # Concatenate
    g = pd.concat((serie,sp500), axis=1)

    # Compute the beta 
    beta = np.cov(g[[serie.name, "SP500"]].dropna().values, rowvar=False)[0][1] / np.var(g["SP500"].dropna().values)
    return beta 
# Create a drawdown function 
def drawdown_function(serie):

    # Compute cumsum of the returns
    cum = serie.dropna().cumsum() + 1

    # Compute max of the cumsum on the period (accumalate max)
    running_max = np.maximum.accumulate(cum)

    # Compute drawdown
    drawdown = cum/running_max - 1
    return drawdown

# Creat a backtesting function to evaluate the strategy 
def BackTest(serie):

    """ Initialisation"""
    # Import the benchmark 
    sp500 = yf.download("^GSPC")["Adj Close"].pct_change(1)

    # Change the name
    sp500.name = "SP500"

    # Concat the return and the sp500
    val = pd.concat((return_rsi_strategy, sp500), axis=1).dropna()

    # Compute the drawdown
    drawdown = drawdown_function(serie)

    # Compute max drawdown
    max_drawdown = -np.min(drawdown) #*100

    """ Plot some graph """
    fig, (cum, dra) = plt.subplots(1,2, figsize=(20,6))

    ## Add a subtitle
    fig.suptitle("Backtesting", size=20)

    # Return cumsun chart
    cum.plot(serie.cumsum(), color="#39B3C7")

    # SP500 cumsum chart
    cum.plot(val["SP500"].cumsum(), color="#B85A0F")

    # Put a legend
    cum.legend(["Portfolio", "Benckmark"])

    # Set indivdual title
    cum.set_title("Cumulative Returns",size=13)

    # Put the drawdown
    dra.fill_between(drawdown.index, 0, drawdown, color="#C73954", alpha=0.65)

    # Set the individual title 
    dra.set_title("Drawdown", size=13)

    # Show the graph
    plt.show()

    # Compute the sortino
    sortino = np.sqrt(252) * serie.mean()/serie.loc[serie<0].std()

    # Compute the beta 
    beta = np.cov(val[["return", "SP500"]].values,rowvar=False)[0][1] / np.var(val["SP500"].values)

    # Compute the alpha
    alpha = 252 * serie.mean() - 252*beta*serie.mean()

    # Print the statistics
    print(f"Sortino: {np.round(sortino,3)}")
    print(f"Beta: {np.round(beta,3)}")
    print(f"Alpha: {np.round(alpha, 3)}")
    print(f"MaxDrawdown: {np.round(max_drawdown*100,3)} %")

## Run the RSI function
if __name__ == "__main__":

    # # Calculate the return of the RSI strstegy
    # return_rsi_strategy = RSI(f,5,14)

    # # Compute drawdown 
    # drawdown = drawdown_function(return_rsi_strategy)

    # # Run backtest
    # BackTest(return_rsi_strategy)

    """
    Hurst Componenet
        - 0.5 < Hurst < 1: Trending movement
        - 0.5 = Hurst: Randdom Walk
        - 0 < Hurst < 0.5: Antipersitent movement
    """

    # # Compute Hurst Exponent
    # # Trending 
    # arr = np.linspace(0, 300, 150) + 100
    # hurst = compute_Hc(arr)[0]

    # # Show the result
    # plt.plot(arr)
    # plt.title(f"{'%.2f' % hurst}")
    # plt.show()

    # # Antipersistent
    # arr = np.cos(np.linspace(0, 300, 150) + 100)
    # hurst = compute_Hc(arr)[0]

    # # Show the result
    # plt.plot(arr)
    # plt.title(f"{'%.2f' % hurst}")
    # plt.show()

    # # Random Walk
    # np.random.seed(56)
    # arr = np.cumsum(np.random.randn(150))
    # hurst = compute_Hc(arr)[0]

    # # Show the result
    # plt.plot(arr)
    # plt.title(f"{'%.2f' % hurst}")
    # plt.show()

    # # Download Name.csv - ticker list 
    # assets = pd.read_csv("Names.csv")["Symbol"]

    # # Initialise our lists
    # Statistics = []
    # col =[]

    # for fin in tqdm(assets):

    #     # Get dataset
    #     try:
    #         print(fin)

    #         # Download data for each asset
    #         f = yf.download(fin).dropna()

    #         # Create a list to put the following statistics
    #         statistics = list()

    #         # Compute the Hurst
    #         statistics.append(compute_Hc(f["Adj Close"])[0])

    #         # Compute the volatitity
    #         statistics.append(np.sqrt(252)*f["Adj Close"].pct_change().std())

    #         # Compute the beta
    #         statistics.append(beta_function(f["Adj Close"].pct_change().dropna()))

    #         # Compute statategy return
    #         statistics.append(RSI(f,5,14).mean()*252)

    #         # Put statistics list in Statistics -> have list of lists
    #         Statistics.append(statistics)

    #         # Put columns name in the list because some columns dont have 100 values
    #         col.append(fin)
        
    #     # If the assets has not 100 values we pass to the next 
    #     except:
    #         pass
            
    # # Create dataframe with all the previous statistics
    # resume = pd.DataFrame(Statistics, columns=["Hurst", "Volatility", "Beta", "Sum Strategy Returns"], index=col)
    # resume.to_csv("resume.csv")
    # print(resume)

    # Get Statistics data
    resume = pd.read_csv("resume.csv",index_col=0)

    # Extract class of the active ticker
    # print(resume)

    # Extract asset type
    clustering = pd.read_csv("Names.csv", index_col="Symbol")
    del clustering["Unnamed: 0"]

    # Concat resume & clustering
    g = pd.concat([resume, clustering], axis=1).dropna()
    print(g)

    # Plot the densities
    sns.displot(data=g, x="Sum Strategy Returns", kind="kde", hue="dummy")

    # Limit the axis 
    plt.xlim((-1.15, 1.15))

    #plot the graph
    plt.show()

    # # Descrive by currency
    # print(g.loc[g["dummy"]=="Currency"].describe())

    # # Descrive by currency
    # print(g.loc[g["dummy"]=="Crypto"].describe())

    # # Descrive by currency
    # print(g.loc[g["dummy"]=="Asset"].describe())

    # Plot the density of the strategy returns by the HURST
    g["Hurst_dum"] = "Low"
    g.loc[g["Hurst"]>0.56, "Hurst_dum"] = "High"

    # Plot the densities
    sns.displot(data=g, x="Sum Strategy Returns", kind="kde", hue="Hurst_dum")

    # Limit the axis 
    plt.xlim((-1.15, 1.15))

    #plot the graph
    plt.show()

    # Plot the density of the strategy returns by the volatility
    g["Volatility_dum"] = "Low"
    g.loc[g["Volatility"]>0.52, "Volatility_dum"] = "High"

    # Plot the densities
    sns.displot(data=g, x="Sum Strategy Returns", kind="kde", hue="Volatility_dum")

    # Limit the axis 
    plt.xlim((-1.15, 1.15))

    #plot the graph
    plt.show()

    # Plot the density of the strategy returns by the beta
    g["Beta_dum"] = "Low"
    g.loc[g["Beta"]>1, "Beta_dum"] = "High"

    # Plot the densities
    sns.displot(data=g, x="Sum Strategy Returns", kind="kde", hue="Beta_dum")

    # Limit the axis 
    plt.xlim((-1.15, 1.15))

    #plot the graph
    plt.show()