import datetime as dt
import pandas as pd
import pytz


class IBDataReader(object):
    def __init__(self, data_path='.'):
        self._data_path = data_path
        self._datetime_format = '%Y%m%d %H:%M:%S'
        self._date_format = '%Y%m%d'

        self._df = None
        self._s = None

    def _reset_data(self):
        self._df = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInterest'])
        self._s = pd.Series()

    def get_dataframe(self, sec_type, symbol, currency, exchange, end_time, duration, bar_size, what_to_show, use_rth):

        self._reset_data()

        filename = self.get_filename(bar_size, currency, duration, end_time, exchange, sec_type, symbol, use_rth,
                                     what_to_show)

        try:
            self._df = pd.read_csv(filename,
                               parse_dates=True,
                               index_col=0)

            # self._df = self._df.sort_index()

            self.filter_out_duplicates(filename)

            if duration == '1 Y':
                year_dt = dt.datetime.strptime(end_time, self._datetime_format)
                self.filter_out_data_not_from_year(filename, year_dt.year)

        except OSError:
            print('Did not find filename: ',filename)

        return self._df

    def get_filename(self, bar_size, currency, duration, end_time, exchange, sec_type, symbol, use_rth, what_to_show):
        # build filename
        file_date = end_time

        if duration == '1 Y':
            year_dt = dt.datetime.strptime(end_time, self._datetime_format)
            file_date = str(year_dt.year)

        filename = self._data_path + '/' + symbol + '_' + sec_type + '_' + exchange + '_' + currency + '_' + \
                   file_date.replace(' ', '') + '_' + duration.replace(' ', '') + '_' + \
                   bar_size.replace(' ', '') + '_' + \
                   what_to_show + '_' + str(use_rth) + '.csv'

        return filename

    def get_multiple_years_of_daily_bars_as_list_of_bt_data_feeds(self, symbol, start_year, end_year, use_rth=1):

        """
        :param symbol:
        :param start_year: integer. Starts returning data
        :param end_year: integer. end year
        :param use_rth: 1 if data from only regular trading hours should be returned.
        0 if before/ after market hours data should be returned.
        :return: A list of pandas dataframes (DF), one per year. Each DF contains columns:
        'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInterest'
        """

        data_list = []

        if start_year == end_year:
            end_year += 1

        for year in range(start_year, end_year + 1):
            df = self.get_a_year_of_daily_bars_as_a_dataframe(symbol, year, use_rth)
            data = bt.feeds.PandasData(dataname=df, tz=pytz.timezone('US/Eastern'))
            data_list.append(data)

        return data_list

    def get_a_year_of_daily_bars_as_a_dataframe(self, symbol, year, use_rth=1):

        end_time = dt.datetime(year, 12, 31, 23, 59, 59).strftime(self._datetime_format)
        # end_time = dt.datetime(year+1, 1, 1).strftime(self._datetime_format)
        # end_time = dt.datetime(year, 1, 1).strftime(self._datetime_format)

        stock_kwargs = dict(
            sec_type='STK',
            symbol=symbol,
            currency='USD',
            exchange='SMART',
            end_time=end_time,
            duration='1 Y',
            bar_size='1 day',
            what_to_show='TRADES',
            use_rth=use_rth
        )

        df = self.get_dataframe(**stock_kwargs)

        # Filer out any data not from this year
        self._df = df[dt.datetime(year, 1, 1):dt.datetime(year, 12, 31)]

        # TODO
        # Detect and filter duplicate dates

        return self._df

    def get_multiple_years_of_daily_bars_as_pandas_dataframe(self, symbol, start_year, end_year, use_rth):

        if start_year == end_year:
            end_year += 1

        stocks = []

        for year in range(start_year, end_year + 1):
            stock_df = self.get_a_year_of_daily_bars_as_a_dataframe(symbol, year, use_rth)
            # stock_df.tz_localize(tz='America/New_York')
            # stock_df = stock_df.sort_index()
            if not stock_df.empty:
                stocks.append(stock_df)

        # stocks_df = pd.concat(stocks, keys=[symbol])
        stocks_df = pd.concat(stocks)
        # stocks_df.tz_localize(tz='America/New_York')
        # stocks_df = stocks_df.sort_index()
        # print(stocks)

        return stocks_df

    def get_multiple_years_of_daily_bars_for_multiple_stocks_as_midf(self,
                                                                     symbols, keys, start_year, end_year, use_rth=1):

        '''
                :param symbols:
                :param keys: One or more columns from an IB datafile:
                Column options are: 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInterest'
                :param use_rth:
                :return: a multi-index dataframe where the major key is a datetime and the minor key is a stock symbol.
                The columns are the keys passed in.
                Using the midf:
                df = stocks_midf.loc['SPY']['close']
                '''

        keys = ['Date'] + keys
        stocks = []
        for symbol in symbols:
            stock_df = self.get_multiple_years_of_daily_bars_as_pandas_dataframe(
                symbol, start_year, end_year, use_rth)
            # stock_df.tz_localize(tz='America/New_York')
            # stock_df = stock_df.sort_index()
            if not stock_df.empty:
                #print('symbol: {} len: {}'.format(symbol, len(stock_df)))
                stocks.append(stock_df)
            # print(stock_df)

        stocks_midf = pd.concat(stocks, keys=symbols, axis=0, join='outer')  # .reset_index()

        if stocks_midf.isnull().values.any():
            raise ValueError('get_stock_data: null values somewhere in dataframe')

        return stocks_midf

    def get_multiple_days_of_minute_bars_as_list_of_bt_data_feeds(self, symbol, date_list, use_rth=1):

        """
        :param symbol:
        :param date_list: list. list of datetimes
        :param use_rth: 1 if data from only regular trading hours should be returned.
        0 if before/ after market hours data should be returned.
        :return: A list of pandas dataframes (DF), one per year. Each DF contains columns:
        'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInterest'
        """

        data_list = []

        for date in date_list:
            print("calling get_a_date for: {}".format(date))
            df = self.get_a_day_of_minute_bars_as_a_dataframe(symbol, date.year, date.month, date.day, use_rth)
            data = bt.feeds.PandasData(dataname=df, tz=pytz.timezone('US/Eastern'))
            data_list.append(data)

        return data_list

    def get_a_day_of_minute_bars_as_a_dataframe(self, symbol, year_int, month_int, day_int, use_rth=1):

        end_time = dt.datetime(year_int, month_int, day_int).strftime(self._datetime_format)

        # Get data for the 14th and 15th (including after hours trading)
        stock_kwargs = dict(
            sec_type='STK',
            symbol=symbol,
            currency='USD',
            exchange='SMART',
            end_time=end_time,
            duration='1 D',
            bar_size='1 min',
            what_to_show='TRADES',
            use_rth=use_rth
        )
        df = self.get_dataframe(**stock_kwargs)
        return df

    def filter_out_duplicates(self, filename, remove=True):
        # Check for duplicate dates
        duplicates_df = self._df[self._df.index.duplicated(keep=False)]
        if not duplicates_df.equals(self._df) and not duplicates_df.empty:
            if remove:
                print("Dataframe has duplicate rows for the same date... removed. Cache file: {}".format(filename))
                self._df.drop_duplicates(inplace=True)
            else:
                print("Dataframe has duplicate rows for the same date. Cache file: {}".format(filename))

    def filter_out_data_not_from_year(self, filename, year_int, remove=True):
        # Check for data outside the year you wanted it in.
        data_from_year_df = self._df[dt.datetime(year_int, 1, 1): dt.datetime(year_int, 12, 31)]
        if not data_from_year_df.equals(self._df) and not self._df.empty:
            # Print extraneous data
            #not_from_year_df = self._df[~self._df.isin(data_from_year_df)].dropna()
            #print(not_from_year_df)
            if remove:
                print("Dataframe has data from not this year... removed. Cache file: {}".format(filename))
                self._df = data_from_year_df
            else:
                print("Dataframe has data from not this year. Cache file: {}".format(filename))
