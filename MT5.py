import warnings
from datetime import datetime
import pandas as pd
import MetaTrader5 as mt5
warnings.filterwarnings("ignore")
mt5.initialize()

class MT5:

    def get_data(symbol, n, timeframe=mt5.TIMEFRAME_D1):
        """ Function to import the data of the chosen symbol"""
        print(symbol)

        # Initialise the connection if there is not
        mt5.initialize()

        # Current date extract
        utc_from = datetime.now()


        # Import the data into a tuple
        rates = mt5.copy_rates_from(symbol, timeframe, utc_from, n)

        # Tuple to dataframe
        rates_frame = pd.DataFrame(rates)

        # Convert time in seconds into the datetime format
        rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')

        # Convert time in seconds into the right format
        rates_frame['time'] = pd.to_datetime(rates_frame['time'], format="%Y-%m-%d")

        # Set column time as the index of the dataframe
        rates_frame = rates_frame.set_index('time')

        return rates_frame

        


if __name__ == "__main__":
    rate_frame = MT5.get_data("NAS100",10)
    print(rate_frame)
