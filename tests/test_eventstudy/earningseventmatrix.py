from eventstudy.eventstudy import *


class EarningsEventMatrix(EventMatrix):
    def __init__(self, datetimes, symbols, value_threshold, positive, csv_file_name):
        EventMatrix.__init__(self, datetimes, symbols)
        self.value_threshold = value_threshold
        self.csv_file_name = csv_file_name
        self.positive = positive
        #print(self.event_matrix.index)

    def build_event_matrix(self, start_date, end_date):
        """
        :param datetimes:
        :param symbols:
        Returns A pandas dataframe indexed by datetimes and with columns for each symbol.
        Each cell contains a 1 if there was an event for that datetime for that security (and nan otherwise).
        """

        # Now get the raw events from a csv
        raw_event_dates = pd.read_csv(self.csv_file_name,
                                      encoding='ISO-8859-1',
                                      usecols=['asof_date', 'symbol', 'eps_pct_diff_surp'],
                                      parse_dates=['asof_date'],
                                      index_col='asof_date')

        # Sort by date
        raw_event_dates = raw_event_dates.sort_index()
        #print(raw_event_dates)
        #print(raw_event_dates.loc['2017':'2018'])

        # Select between certain dates.
        raw_event_dates = raw_event_dates.loc[start_date:end_date]

        if self.positive:
            raw_event_dates = raw_event_dates[raw_event_dates['eps_pct_diff_surp'] >= self.value_threshold]
        else:
            raw_event_dates = raw_event_dates[raw_event_dates['eps_pct_diff_surp'] <= self.value_threshold]

        #print(raw_event_dates)
        #print(raw_event_dates.loc['2001':'2002'])

        # for each event, put a "1" in the matrix on that day. If the event didn't fall on a trading day,
        # put the event on the first trading day after.
        for index, row in raw_event_dates.iterrows():
            # find the trading day equal to it or after it
            index_df = self.event_matrix.index.get_loc(index, method="backfill")
            index_str = self.event_matrix.index[index_df]

            # Mark the event on the symbol
            self.event_matrix.loc[index_str, row['symbol']] = 1

        return self.event_matrix
