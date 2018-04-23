import pandas as pd


class AVDataCache(object):

    def __init__(self, data_dir):
        self.__data_dir = data_dir
        return

    def get_stock_data(self, symbols, keys):
        '''
        :param symbols:
        :param keys:
        :return: a multi-index dataframe where the major key is a datetime and the minor key is a stock symbol.
        The columns are the keys passed in.
        Column options are: open,high,low,close,adjusted_close,volume,dividend_amount,split_coefficient
        a pandas dataframe with datetime indices and columns of stock symbols.
        Using the midf:
        df = stocks_midf.loc['SPY']['close']
        '''

        keys = ['timestamp'] + keys
        stocks = []
        for symbol in symbols:
            file = self.__data_dir + 'daily_adjusted_' + symbol + '.csv'
            stock_df = pd.read_csv(file, usecols=keys, parse_dates=True, index_col=0)
            stock_df.tz_localize(tz='America/New_York')
            stock_df = stock_df.sort_index()
            stocks.append(stock_df)

        stocks_midf = pd.concat(stocks, keys=symbols)
        # print(stocks_midf.loc['SPY']['close'])

        if stocks_midf.isnull().values.any():
            raise ValueError('get_stock_data: null values somewhere in dataframe')

        return stocks_midf

