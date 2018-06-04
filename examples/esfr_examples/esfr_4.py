import pandas as pd
import numpy as np
from io import StringIO
from eventstudy.naivemodel import EventStudyNaiveModel
from eventstudy.dataprovider import DataProvider


def build_events():

    """Return a pandas dataframe of the events (since the csv files always get deleted.)"""
    df = pd.DataFrame(columns=['event_date', 'day_0_date', 'ticker'])

    event_string = """1/31/2000,2/1/2000,ALK
11/12/2001,11/12/2001,AAMRQ
4/1/2011,4/2/2011,LUV
8/14/2013,8/14/2013,UPS"""

    event_list = event_string.splitlines()

    for row in event_list:
        event = row.split(',')
        df = df.append([{'event_date': event[0], 'day_0_date': event[1], 'ticker': event[2]}],
                       ignore_index=True)

    # convert date strings to datetimes
    df['day_0_date'] = pd.to_datetime(df['day_0_date'])

    return df

class DataProviderESFR(DataProvider):

    """Provide security price data specifically for this event study."""

    def __init__(self):
        pass

    def get_closing_prices(self, ticker, day_0_date, num_pre_event_window_periods, num_post_event_window_periods):

        """Return a pandas DataFrame of closing prices over the event window."""

        columns = self.get_event_window_columns(num_pre_event_window_periods, num_post_event_window_periods)
        closing_prices_df = pd.DataFrame(index=[0], columns=columns)

        # Create a list of prices for the event window and fill with NaN
        prices = [np.nan for i in range(num_pre_event_window_periods + 1 + num_post_event_window_periods)]

        if ticker == 'ALK':
            prices = [14.97, 15.16, 16.03, 15.84, 16.22, 15.84, 15.78, 15.81, 15.47, 15.25, 15.10, 14.94, 14.70]
        elif ticker == 'AAMRQ':
            prices = [18.74, 18.90, 18.36, 18.55, 18.28, 18.13, 16.49, 17.01, 17.94, 18.75, 20.06, 20.77, 19.86]
        elif ticker == 'LUV':
            prices = [12.53, 12.43, 12.36, 12.66, 12.48, 12.52, 12.31, 12.05, 12.01, 11.66, 11.54, 11.64, 11.70]
        elif ticker == 'UPS':
            prices = [86.79, 86.72, 86.96, 86.65, 86.61, 86.55, 86.30, 85.43, 85.52, 85.55, 85.77, 85.56, 86.43]
        elif ticker == 'SPY':
            if day_0_date.strftime('%Y-%m-%d') == '2000-02-01':
                prices = [1401.91, 1410.03, 1404.09, 1398.56, 1360.16, 1394.46, 1409.28, 1409.12, 1424.97, 1424.37,
                          1423.00, 1441.75, 1411.71]
            elif day_0_date.strftime('%Y-%m-%d') == '2001-11-12':
                prices = [1087.20, 1102.84, 1118.86, 1115.80, 1118.54, 1120.31, 1118.33, 1139.09, 1141.21, 1142.24,
                          1138.65, 1151.06, 1142.66]
            elif day_0_date.strftime('%Y-%m-%d') == '2011-04-02':
                prices = [1313.80, 1310.19, 1319.44, 1328.26, 1325.83, 1332.41, 1332.87, 1332.63, 1335.54, 1333.51,
                          1328.17, 1324.46, 1314.16]
            elif day_0_date.strftime('%Y-%m-%d') == '2013-08-14':
                prices = [1697.37, 1690.91, 1697.48, 1691.42, 1689.47, 1694.16, 1685.39, 1661.32, 1655.83, 1646.06,
                          1652.35, 1642.80, 1656.96]

        closing_prices_df.loc[0] = prices
        return closing_prices_df


def load_events():
    """Pretend to read a csv and return a list of events as a pandas DataFrame."""

    event_data = StringIO("""event_date,day_0_date,ticker
1/31/2000,2/1/2000,ALK
11/12/2001,11/12/2001,AAMRQ
4/1/2011,4/2/2011,LUV
8/14/2013,8/14/2013,UPS
        """)

    event_list_df = pd.read_csv(event_data,
                                usecols=['day_0_date', 'ticker'],
                                parse_dates = ['day_0_date'],
                                )

    return event_list_df


def main():

    """
    This example replicates "Event Studies for Financial Research, chapter 4:
    A simplified example, the effect of air crashes on stock prices"
    using the eventstudy package.
    """

    event_list_df = build_events()
    # print('The event list:\n {}'.format(event_list_df))

    data_provider = DataProviderESFR()

    event_study = EventStudyNaiveModel(data_provider, event_list_df)

    # Run the event study looking 6 periods before the event and 6 periods after the event
    num_pre_event_window_periods = num_post_event_window_periods = 6
    market_ticker = 'SPY'
    results = event_study.run_naive_model(market_ticker, num_pre_event_window_periods, num_post_event_window_periods)

    print('\nStarted with {} events and processed {} events.'.format(results.num_starting_events,
                                                                    results.num_events_processed))

    print('\nAAR (%) for all the securities over the event window:\n{}'.format(
        (results.aar * 100).round(2).to_frame().T.to_string(index=False)))

    print('\nCAAR (%) for all the securities over the event window:\n{}'.format(
        (results.caar * 100).round(2).to_frame().T.to_string(index=False)))

    #print('aar t-statistic: {}'.format(results.aar_test_statistic))
    #print('aar p-val: {}'.format(results.aar_p_val))
    #print('caar t-statistic: {}'.format(results.caar_test_statistic))
    #print('caar p-val: {}'.format(results.caar_p_val))

    results.plot("Airline Crashes and their impact on stock returns", False, 'esfr_4.pdf')


if __name__ == '__main__':
    main()
