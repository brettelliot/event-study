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
        data_df = pdr.get_data_yahoo("SPY", start="2017-01-01", end="2017-04-30", session=session)

        if data_df.isnull().values.any():
            return closing_prices_df

        print(data_df.head())
        return closing_prices_df
