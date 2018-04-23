from datetime import datetime
import pandas as pd
#import pytz, tzlocal

class IBDataCache(object):

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

        # build filename
        filename = self._data_path + '/' + symbol + '_' + sec_type + '_' + exchange + '_' + currency + '_' + \
                   end_time.replace(' ', '') + '_' + duration.replace(' ', '') + '_' + bar_size.replace(' ', '') + '_' + \
                   what_to_show + '_' + str(use_rth) + '.csv'

        self._df = pd.read_csv(filename,
                               parse_dates=True,
                               index_col=0)

        # self._df = self._df.sort_index()

        return self._df

    def get_a_year_of_daily_bars_as_a_dataframe(self, symbol, year, use_rth):

        end_time = datetime(year, 12, 31).strftime(self._datetime_format)

        stock_kwargs = dict(
            sec_type='STK',
            symbol=symbol,
            currency='USD',
            exchange='SMART',
            # The current implementation gets from 12/31 of the previous year. Test this to see if it gets 1/1
            # endtime=datetime(year+1, 1, 1).strftime(date_format),
            end_time=end_time,
            duration='1 Y',
            bar_size='1 day',
            what_to_show='TRADES',
            use_rth=use_rth
        )
        df = self.get_dataframe(**stock_kwargs)
        return df

    def get_multiple_years_of_daily_bars_as_pandas_dataframe(self, symbol, start_year, end_year, use_rth):

        if start_year == end_year:
            end_year += 1

        stocks = []

        for year in range(start_year, end_year+1):
            stock_df = self.get_a_year_of_daily_bars_as_a_dataframe(symbol, year, use_rth)
            #stock_df.tz_localize(tz='America/New_York')
            #stock_df = stock_df.sort_index()
            stocks.append(stock_df)

        #stocks_df = pd.concat(stocks, keys=[symbol])
        stocks_df = pd.concat(stocks)
        #stocks_df.tz_localize(tz='America/New_York')
        #stocks_df = stocks_df.sort_index()
        #print(stocks)

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
            #stock_df.tz_localize(tz='America/New_York')
            #stock_df = stock_df.sort_index()
            stocks.append(stock_df)
            #print(stock_df)

        stocks_midf = pd.concat(stocks, keys=symbols, axis=0, join='outer')#.reset_index()

        if stocks_midf.isnull().values.any():
            raise ValueError('get_stock_data: null values somewhere in dataframe')

        return stocks_midf



