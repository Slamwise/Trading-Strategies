import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time

def get_minute_bars(ticker):
    """
    Fetch minute-by-minute price data for a given stock ticker as far back as yahoo finance allows.

    Args:
    ticker (str): The stock ticker symbol.

    Returns:
    pandas.DataFrame: A DataFrame containing the minute-by-minute price data.
    """

    # Initialize the ticker object
    stock = yf.Ticker(ticker)

    # Set the end date to now and initialize an empty DataFrame for all data
    end_date = datetime.now()
    all_data = pd.DataFrame()

    # Function to fetch data with error handling
    def fetch_data(ticker, start, end, interval='1m'):
        try:
            df = ticker.history(start=start, end=end, interval=interval)
            if df.empty:
                raise ValueError("No data returned")
            return df
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return pd.DataFrame()

    # Fetch data in 7-day intervals
    while True:
        start_date = end_date - timedelta(days=7)
        data = fetch_data(stock, start_date, end_date)
        
        if data.empty:
            break
        
        all_data = pd.concat([data, all_data])
        end_date = start_date
        
        print(f"Fetched data from {start_date} to {end_date}")
        
        # Add a 1 second sleep between requests
        time.sleep(0.5)

    if not all_data.empty:
        print(f"\nData retrieved for {ticker}")
        print(f"Total number of minutes of data: {len(all_data)}")
        print(f"Date range: {all_data.index[-1]} to {all_data.index[0]}")
        return all_data
    else:
        print(f"No data available for {ticker}.")
        return None
