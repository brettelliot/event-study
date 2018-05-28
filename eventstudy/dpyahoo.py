from eventstudy.dataprovider import DataProvider
import datetime as dt
import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import requests_cache

import fix_yahoo_finance as yf
yf.pdr_override()

class DataProviderYahoo(DataProvider):

    """Provide security price data specifically for this event study."""

    def __init__(self):
        DataProvider.__init__(self)
        pass

    def get_closing_prices(self, ticker, day_0_date, num_pre_event_window_periods, num_post_event_window_periods):

        """Return a pandas DataFrame of closing prices over the event window."""

        # prep the return value
        columns = self.get_event_window_columns(num_pre_event_window_periods, num_post_event_window_periods)
        closing_prices_df = pd.DataFrame(index=[0], columns=columns)
        prices = [np.nan for i in range(num_pre_event_window_periods + 1 + num_post_event_window_periods)]
        closing_prices_df.loc[0] = prices

        # The periods are trading days, but yahoo requires start and end dates.
        # We want to make sure we cover the event window so be conservative with our data pulling here so
        # triple it.
        pre_window_delta = num_post_event_window_periods * 3
        post_window_delta = num_post_event_window_periods * 3

        start_date = day_0_date - dt.timedelta(days=pre_window_delta)
        end_date = day_0_date + dt.timedelta(days=post_window_delta)
        # print('day_0_date: {}, start_date: {}, end_date: {}'.format(day_0_date, start_date, end_date))

        # Get a pandas dataframe of closing prices
        session = requests_cache.CachedSession(cache_name='cache', backend='sqlite')
        data_df = pdr.get_data_yahoo(ticker, start=start_date, end=end_date, session=session)

        # Return an empty dataframe if we don't get any data
        if data_df.isnull().values.any():
            return closing_prices_df
        elif data_df.empty:
            return closing_prices_df

        # Find the index of the event at day 0
        index = data_df.index.searchsorted(day_0_date)

        # Get the prices for the event window
        event_window_start = index-num_pre_event_window_periods
        event_window_end = index+num_post_event_window_periods+1
        prices = data_df.iloc[event_window_start:event_window_end]['Adj Close']

        # Set them in the dataframe we are returning
        closing_prices_df.loc[0] = prices.values

        return closing_prices_df
