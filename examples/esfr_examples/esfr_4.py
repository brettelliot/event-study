import pandas as pd
import numpy as np

class EventStudyDataProvider(object):
    """Class that provides security data specifically for and EventStudy"""

    def __init__(self):
        pass

    def get_closing_prices(self, ticker, day_0_date, num_pre_event_window_periods, num_post_event_window_periods):

        """Return a pandas DataFrame of closing prices over the event window."""

        columns = ['-6', '-5', '-4', '-3', '-2', '-1', '0', '1', '2', '3', '4', '5', '6']
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
            if day_0_date == '2/1/2000':
                prices = [1401.91, 1410.03, 1404.09, 1398.56, 1360.16, 1394.46, 1409.28, 1409.12, 1424.97, 1424.37,
                          1423.00, 1441.75, 1411.71]
            elif day_0_date == '11/12/2001':
                prices = [1087.20, 1102.84, 1118.86, 1115.80, 1118.54, 1120.31, 1118.33, 1139.09, 1141.21, 1142.24,
                          1138.65, 1151.06, 1142.66]
            elif day_0_date == '4/2/2011':
                prices = [1313.80, 1310.19, 1319.44, 1328.26, 1325.83, 1332.41, 1332.87, 1332.63, 1335.54, 1333.51,
                          1328.17, 1324.46, 1314.16]
            elif day_0_date == '8/14/2013':
                prices = [1697.37, 1690.91, 1697.48, 1691.42, 1689.47, 1694.16, 1685.39, 1661.32, 1655.83, 1646.06,
                          1652.35, 1642.80, 1656.96]

        closing_prices_df.loc[0] = prices
        return closing_prices_df


class EventStudyResults(object):
    """Helper class that collects and formats Event Study Results."""

    def __init__(self):
        self.caar = []


class EventStudy(object):
    """Tool that runs an event study, calculating CAAR using the naive benchmark model."""

    def __init__(self, data_provider, event_list_df):
        """Create an event study.

        Args:
            data_provider (:obj:`EventStudyDataProvider`): object that gets security data
            event_list_df (:obj:`DataFrame`): Pandas DataFrame containing the list of event dates and ticker symbols
        """

        self.data_provider = data_provider
        self.event_list_df = event_list_df
        self.num_pre_event_window_periods = 0
        self.num_post_event_window_periods = 0
        self.results = None

        pass

    def run_naive_model(self, market_ticker, num_pre_event_window_periods, num_post_event_window_periods):
        """Run the event study using the naive benchmark model and return the results.

        Args:
            market_ticker (str): The ticker of the model's benchmark
            num_pre_event_window_periods (int): The number of periods before the event
            num_post_event_window_periods (int): The number of periods after the event

        Returns:
            An instance of EventStudyResults.

        """

        self.results = EventStudyResults()
        self.num_pre_event_window_periods = num_pre_event_window_periods
        self.num_post_event_window_periods = num_post_event_window_periods

        for index, event in self.event_list_df.iterrows():

            """
            For each event:
                Get the closing prices of the securities for the event window
                Get the closing prices of the market benchamark for the event window
                Calculate the returns for the security.
                Calculate the and market around the event window
            """

            print('\nDay Zero Date: {} ticker: {}'.format(event.day_0_date, event.ticker))

            # Get prices for the security over the event window
            security_prices_df = self.data_provider.get_closing_prices(event.ticker,
                                                                      event.day_0_date,
                                                                      self.num_pre_event_window_periods,
                                                                      self.num_post_event_window_periods)

            if security_prices_df.isnull().values.any():
                print('\n**** Prices for {} are missing  around date: {} ****'.format(event.ticker, event.day_0_date))
                continue

            # Get prices for the market benchmark over the event window
            market_prices_df = self.data_provider.get_closing_prices(market_ticker,
                                                                    event.day_0_date,
                                                                    self.num_pre_event_window_periods,
                                                                    self.num_post_event_window_periods)

            if market_prices_df.isnull().values.any():
                print('\n**** Prices for {} are missing around date: {} ****'.format(market_ticker, event.day_0_date))
                continue

            # print('\n{} Security prices($):\n{}'.format(event.ticker, security_prices_df.to_string(index=False)))
            #print('\n{} Market prices($):\n{}'.format(market_ticker, market_prices_df.to_string(index=False)))

            # Calculate the arithmetic return for the security over the event window
            security_returns_df = security_prices_df.pct_change(axis='columns')
            #print('\nSecurity Returns(%) for {}:\n{}'.format(event.ticker,(security_returns_df*100).round(2).to_string(index=False)))

            # Calculate the arithmetic return for the market over the event window.
            # In the naive model, this becomes the Normal Return.
            normal_returns_df = market_prices_df.pct_change(axis='columns')
            #print('\nNormal Returns(%) for {}:\n{}'.format(market_ticker,(normal_returns_df*100).round(2).to_string(index=False)))

            # Calculate the Abnormal Return over the event window
            # AR = Stock Return - Normal Return
            abnormal_returns_df = security_returns_df.sub(normal_returns_df)
            print('\nAbnormal Returns(%) for {}:\n{}'.format(event.ticker,(abnormal_returns_df*100).round(2).to_string(index=False)))

        return self.results


def load_events(file_name):
    """Read a csv and return a list of events as a pandas DataFrame."""

    event_list_df = pd.read_csv(file_name,
                                encoding='ISO-8859-1',
                                usecols=['day_0_date', 'ticker'],
                                # parse_dates=['day_0_date'],
                                # index_col='day_0_date',
                                )

    return event_list_df


def main():
    """
    This example is from "Event Studies for Financial Research, chapter 4:
    A simplified example, the effect of air crashes on stock prices
    :return:
    """

    event_list_df = load_events('esfr_4_events.csv')
    # print('The event list:\n {}'.format(event_list_df))

    data_provider = EventStudyDataProvider()

    event_study = EventStudy(data_provider, event_list_df)

    # Run the event study looking 6 periods before the event and 6 periods after the event
    num_pre_event_window_periods = num_post_event_window_periods = 6
    market_ticker = 'SPY'
    results = event_study.run_naive_model(market_ticker, num_pre_event_window_periods, num_post_event_window_periods)

    print(results.caar)


if __name__ == '__main__':
    main()
