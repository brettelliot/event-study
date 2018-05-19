from eventstudy.eventstudy import *


class TestCARSEventMatric(EventMatrix):
    def __init__(self, datetimes, symbols, market_symbol, stock_data, daily_diff, positive):
        EventMatrix.__init__(self, datetimes, symbols)
        self.market_symbol = market_symbol
        self.stock_data = stock_data
        self.daily_diff = daily_diff
        self.positive = positive

    def build_event_matrix(self, start_date, end_date):
        '''
        :param datetimes:
        :param symbols:
        Returns A pandas dataframe indexed by datetimes and with columns for each symbol.
        Each cell contains a 1 if there was an event for that datetime for that security (and nan otherwise).
        '''

        # Convert prices into daily returns.
        # This is the amount that the specific stock increased or decreased in value for one day.
        daily_returns = self.stock_data['Close'].copy()
        daily_returns = daily_returns.pct_change().fillna(0)

        # Create an events data frame data_events, where columns = names of all stocks, and rows = daily dates
        #events_col = symbols_list[:]  # Use [:] to deep copy the list
        #events_col.remove('^GSPC')  # We dont't need to create events for the S&P500
        #events_index = data_ret.index  # Copy the date index from data_ret to the events data frame
        #data_events = pd.DataFrame(index=events_index, columns=events_col)

        # Fill in data_events with 1 for positive events, -1 for negative events, and NA otherwise.
        for i in self.symbols:
            if self.positive:
                self.event_matrix[i] = np.where((daily_returns[i] -
                                             daily_returns[self.market_symbol]) >= self.daily_diff, int(1), np.nan)
            else:
                self.event_matrix[i] = np.where((daily_returns[i] -
                                                 daily_returns[self.market_symbol]) <= self.daily_diff, int(1), np.nan)

        return self.event_matrix
