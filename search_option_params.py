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
    val = pd.concat((serie, sp500), axis=1).dropna()

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

# Function - Heatmap Optimization Parameters
def grid_parameters(f):
    # Set list for the possible values of neutral and window
    neutral_values = [i*2 for i in range(10)]
    window_values = [i*2 for i in range(1,11)]

    # Set the matrix with only zeros
    grid = np.zeros([len(neutral_values), len(window_values)])

    # Calculate the return of the strategy for each combinations
    for i in range(len(neutral_values)):
        for j in range(len(window_values)):

            # Compute return strategy
            return_rsi = RSI(f, neutral_values[i], window_values[j])

            # Compute annualized sortino
            grid[i][j]= np.sqrt(252) * return_rsi.mean()/ (return_rsi[return_rsi<0].std() + 0.00001)

    return grid

## Run the RSI function
if __name__ == "__main__":
    f = yf.download("GOOG")

    # # Adapt the size
    # plt.figure(figsize=(15,8))

    # Palette for color
    pal = sns.color_palette("light:#5A9", as_cmap=True)

    # Find heatmap optimisation
    grid = grid_parameters(f)

    neutral_values = [i*2 for i in range(10)]
    window_values = [i*2 for i in range(1, 11)]

    # Set some datasets
    start_train, end_train = "2017-01-01", "2019-01-01"
    start_test, end_test = "2019-01-01", "2020-01-01"
    start_valid, end_valid = "2020-01-01", "2021-01-01"

    # Create the grids
    grid_train = grid_parameters(f.loc[start_train:end_train])
    grid_test = grid_parameters(f.loc[start_test:end_test])

    # Create a subplot
    fig, (train, test) = plt.subplots(1,2,figsize=(30,6))

    # Add a sup title 
    fig.suptitle("Optimization parameters RSI")

    # Change the color 
    pal = sns.color_palette("light:#5A9", as_cmap=True)

    # Train
    # Add train heatmap
    sns.heatmap(grid_train, annot=True, ax=train, xticklabels=neutral_values, yticklabels=window_values, cmap=pal) #  

    # Add a title 
    train.set_title("Train")

    # Add a xlabel
    train.set_xlabel("Neutral")

    # Put a ylabel
    train.set_ylabel("Window")

    # Test
    # Add train heatmap
    sns.heatmap(grid_test, annot=True, ax=test, xticklabels=neutral_values, yticklabels=window_values, cmap=pal) #  

    # Add a title 
    test.set_title("Test")

    # Add a xlabel
    test.set_xlabel("Neutral")

    # Put a ylabel
    test.set_ylabel("Window")
    
    # Valid
    # return_rsi_strategy = RSI(f.loc["2010"],10,3)
    BackTest(RSI(f.loc["2010"],10,3))

    # # Show graph
    plt.show()

    