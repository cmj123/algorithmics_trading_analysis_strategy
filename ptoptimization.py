# Import library
import numpy as np
from numpy.core.fromnumeric import mean
from scipy import optimize
from scipy.optimize import minimize
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
import os.path
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
def BackTest(serie, year=False):

    """ Initialisation"""
    # Import the benchmark
    if year:
        sp500 = yf.download("^GSPC")["Adj Close"].loc[str(year)].pct_change(1)
    else:
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
    print(val)
    beta = np.cov(val,rowvar=False)[0][1] / np.var(val["SP500"].dropna())

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

# Create an optimisation function
def opt(f):
    # List for the possible values of neutral and window
    neutral_values = [i*2 for i in range(10)]
    window_values = [i*2 for i in range(1,11)]

    # Set some datasets
    start_train, end_train = "2017-01-01", "2019-01-01"
    start_test, end_test = "2019-01-01", "2020-01-01"
    start_valid, end_valid = "2020-01-01", "2021-01-01"

    # Initialise the list
    resume = []

    # Loop to add the values in the list
    for i in range(len(neutral_values)):
        for j in range(len(window_values)):

            # Compute the returns
            return_train = RSI(f.loc[start_train:end_train], neutral_values[i], window_values[i])
            return_test = RSI(f.loc[start_test:end_test], neutral_values[i], window_values[i])

            # Compute the Sortinon
            sortino_train = np.sqrt(252)*return_train.mean() / return_train[return_train<0].std()
            sortino_test = np.sqrt(252)*return_test.mean()/ return_test[return_test<0].std()

            # Create a list of list to create a result dataframe
            values = [neutral_values[i], window_values[j], sortino_train, sortino_test]
            resume.append(values)

    resume = pd.DataFrame(resume, columns=["Neutral", "Window", "Sortino Train", "Sortino Test"])

    # Order by sortino
    ordered_resume = resume.sort_values(by="Sortino Train", ascending=False)
    # print(ordered_resume)

    for i in range(len(resume)):
        # Take the best
        best = ordered_resume.iloc[0+i:1+i,:]

        # Compute the sortino 
        Strain = best["Sortino Train"].values[0]
        Stest = best["Sortino Test"].values[0]

        # Take best neutral and best window
        best_neutral = best["Neutral"].values[0]
        best_window = best["Window"].values[0]

        # If the Sortino of the train and the test are good we stop the loop
        if Stest >0.5 and Strain > 0.5:
            # print(i)
            break 

        # If ther is not values good enough put 0 in all values
        else:
            best_neutral = ordered_resume.iloc[0,0]
            best_window = ordered_resume.iloc[0,1]
            Strain = ordered_resume.iloc[0,2]
            Stest = ordered_resume.iloc[0,3]
    return [best_neutral, best_window, Strain, Stest]

# Function - Sortinon Ratio Criterion
def SR_criterion(weight, returns):
    '''
    output  - opposite Sortino ratio to do a minimisation
    Inputs  - Weight (type ndarray numpy): Weight of portfolio
            - returns (type dataframe pandas): Returns of stocks
    '''

    pf_return = returns.values.dot(weight)
    mu = np.mean(pf_return)
    sigma = np.std(pf_return[pf_return<0])
    Sortino = -mu/sigma
    return Sortino

## Min Variance Optimization
def MV_criterion(weight, Returns_data):
    """
    Output: optimisation portfolio criterion
    Inputs: weight (type ndarray numpy): Weights for portfolio
            Return_data (type ndarray numpy): Return of stocks
    """

    portfolio_return = np.multiply(Returns_data, np.transpose(weight))
    portfolio_return = np.sum(portfolio_return,1)
    mean_ret = np.mean(portfolio_return,0)
    sd_ret = np.std(portfolio_return, 0)
    criterion = sd_ret
    return criterion

## Run the RSI function
if __name__ == "__main__":
    file_path = "./res.csv"
    if os.path.exists(file_path):
        res =  pd.read_csv(file_path, index_col="Asset")
        # print(res)
    else:
        assets = pd.read_csv("Names.csv")["Symbol"]

        # Intialise the lists
        resume = []
        col = []

        # Compute best parameters for each asset
        for fin in tqdm(assets):
            try:
                # Import the asset
                arr = yf.download(fin)

                # Put the values
                resume.append(opt(arr))
                col.append(fin)
            except:
                pass

        # Add assets columns to each list of resume
        for i in range(len(resume)):
            resume[i].append(col[i])

        # Create a dataframe 
        res = pd.DataFrame(resume, columns=["Neutral", "Window","Train","Test","Asset"])

        # Index by asset
        res = res.set_index("Asset")
        res.to_csv("res.csv")

    # Order the dataframe using the Trian sortino
    values = res.sort_values(by="Train", ascending=False)
    values.dropna(inplace=True)
    # print(value.head(20))

    # Border of sets
    start_train, end_train = "2017-01-01", "2019-01-01"
    start_test, end_test = "2019-01-01", "2020-01-01"
    start_valid, end_valid = "2020-01-01", "2021-01-01"

    # Create a dataframe to put the strategies (The assets of the portfolio)
    strategies = pd.DataFrame()
    for col in values.index[0:15]:
        # Import the asset
        l = yf.download(col)

        # Extract optimal neutral
        best_neutral = values.loc[col]["Neutral"]

        # Extract optimal neutral
        best_window = values.loc[col]["Window"]

        # RSI returns
        strategies[f"{col}"] = RSI(l.loc[start_train:], best_neutral, int(best_window))

    # print(strategies.dropna().head())

    n = len(strategies.transpose())

    x0 = np.zeros(n)+(1/n)

    # Optimisation constraints problem
    cons = ({'type':'eq', 'fun':lambda x:sum(abs(x))-1})

    Bounds = [(0,1) for i in range(0, n)]

    # Optimisation problem solving - sortino
    res_SR = minimize(SR_criterion, x0, method="SLSQP", args=(strategies.loc[start_train:end_test].dropna()), 
                    bounds=Bounds, constraints=cons, options={'disp':False})

    # Result for visualisatiob
    X = res_SR.x
    print(np.round(X, 3))
    # sr = np.multiply(strategies.loc[start_valid:end_valid],X).sum(axis=1)
    # BackTest(sr)

    # Optimisation problem solving - mean - variance
    res_MV = minimize(MV_criterion, x0, method="SLSQP", args=(strategies.loc[start_train:end_test].dropna()), bounds=Bounds, constraints=cons, options={'disp':False})
    X = res_MV.x
    # sr = np.multiply(strategies.loc[start_valid:end_valid],X).sum(axis=1)
    # BackTest(sr)
    print(np.round(X, 3))


    