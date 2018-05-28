import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO


class EventStudyESFR(object):
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
        self.number_events = 0
        self.std_err = None


class EventStudy(object):
    """Tool that runs an event study, calculating CAAR using the naive benchmark model."""

    def __init__(self, data_provider, event_list_df):
        """Create an event study.

        Args:
            data_provider (:obj:`EventStudyESFR`): object that gets security data
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

        # Create a DataFrame to hold all the Abnormal Returns which will be used
        # to calculate the Average Abnormal Return (AAR)
        columns = ['-6', '-5', '-4', '-3', '-2', '-1', '0', '1', '2', '3', '4', '5', '6']
        all_abnormal_returns_df = pd.DataFrame(columns=columns)
        #print('\nAll Abnormal Returns: \n{}'.format(all_abnormal_returns_df))

        for index, event in self.event_list_df.iterrows():

            """
            For each event:
                Get the closing prices of the securities for the event window
                Get the closing prices of the market benchamark for the event window
                Calculate the actual returns for the security.
                Calculate the market returns (ie: normal returns) for the benchmark.
                Calculate the abnormal return.
                Calculate the average abnormal return.
                Calculate the cumulative average abnormal return.
            """

            print('\nDay Zero Date: {} ticker: {}'.format(event.day_0_date, event.ticker))

            # Get prices for the security over the event window
            security_prices_df = self.data_provider.get_closing_prices(event.ticker,
                                                                      event.day_0_date,
                                                                      self.num_pre_event_window_periods,
                                                                      self.num_post_event_window_periods)

            if security_prices_df.isnull().values.any():
                print('\n**** Prices for {} are missing around date: {} ****'.format(event.ticker, event.day_0_date))
                continue

            #print('\nSecurity prices($) for {} over the event window:\n{}'.format(event.ticker, security_prices_df.to_string(index=False)))

            # Get prices for the market benchmark over the event window
            market_prices_df = self.data_provider.get_closing_prices(market_ticker,
                                                                    event.day_0_date,
                                                                    self.num_pre_event_window_periods,
                                                                    self.num_post_event_window_periods)

            if market_prices_df.isnull().values.any():
                print('\n**** Prices for {} are missing around date: {} ****'.format(market_ticker, event.day_0_date))
                continue

            #print('\nMarket prices($) for {} over the event window:\n{}'.format(market_ticker, market_prices_df.to_string(index=False)))

            # Calculate the actual arithmetic return for the security over the event window
            actual_returns_df = security_prices_df.pct_change(axis='columns')
            #print('\nSecurity Returns(%) for {} over the event window::\n{}'.format(event.ticker,(actual_returns_df*100).round(2).to_string(index=False)))

            # Calculate the arithmetic return for the market over the event window.
            # In the naive model, this becomes the Normal Return.
            normal_returns_df = market_prices_df.pct_change(axis='columns')
            #print('\nNormal Returns(%) for {} over the event window:\n{}'.format(market_ticker,(normal_returns_df*100).round(2).to_string(index=False)))

            # Calculate the Abnormal Return over the event window
            # AR = Stock Return - Normal Return
            abnormal_returns_df = actual_returns_df.sub(normal_returns_df)
            #print('\nAbnormal Returns(%) for {} over the event window:\n{}'.format(event.ticker,(abnormal_returns_df*100).round(2).to_string(index=False)))

            # Append the AR to the other ARs so we can calculate AAR later
            all_abnormal_returns_df = pd.concat([all_abnormal_returns_df, abnormal_returns_df], ignore_index=True)


        # Calculate the Average Abnormal Returns (AAR) by adding the AR to the AAR and then taking the mean
        #print('\nAR(%) for all securities over the event window:\n{}'.format((all_abnormal_returns_df*100).round(2)))
        average_abnormal_returns_df = all_abnormal_returns_df.mean().to_frame()
        #print('\nAAR(%) for all the securities over the event window:\n{}'.format((average_abnormal_returns_df*100).round(2).T.to_string(index=False)))

        # Calculate the Cumulative Average Abnormal Returns
        caar = average_abnormal_returns_df.cumsum()
        #print(caar)
        #print('\nCAAR(%) for all the securities over the event window:\n{}'.format((caar * 100).round(2).T.to_string(index=False)))

        self.results.caar = caar
        self.results.number_events = all_abnormal_returns_df.shape[0]

        return self.results


def load_events():
    """Pretend to read a csv and return a list of events as a pandas DataFrame."""

    event_data = StringIO("""event_date,day_0_date,ticker
1/31/2000,2/1/2000,ALK
11/12/2001,11/12/2001,AAMRQ
4/1/2011,4/2/2011,LUV
8/14/2013,8/14/2013,UPS
        """)

    event_list_df = pd.read_csv(event_data,
                                usecols=['day_0_date', 'ticker'])

    return event_list_df


def plot_results(results, show=True, pdf_filename=None):
    # Plot pos_CAR and neg_CAR
    plt.clf()
    plt.grid()
    plt.plot(results.caar * 100, label="N=%s" % results.number_events)
    plt.legend(loc='upper right')
    plt.title("CAAR before and after event")
    plt.ylabel('CAAR (%)')
    plt.xlabel('Window')
    plt.rcParams.update({'font.size': 8})
    if pdf_filename is not None:
        plt.savefig(pdf_filename, format='pdf')
    if show:
        plt.show()


def main():

    """
    This example is from "Event Studies for Financial Research, chapter 4:
    "A simplified example, the effect of air crashes on stock prices"

    It is a simple but complete example of the naive model event study approach.
    """

    event_list_df = load_events()
    # print('The event list:\n {}'.format(event_list_df))

    data_provider = EventStudyESFR()

    event_study = EventStudy(data_provider, event_list_df)

    # Run the event study looking 6 periods before the event and 6 periods after the event
    num_pre_event_window_periods = num_post_event_window_periods = 6
    market_ticker = 'SPY'
    results = event_study.run_naive_model(market_ticker, num_pre_event_window_periods, num_post_event_window_periods)

    print('\nCAAR (%) for all the securities over the event window:\n{}'.format(
        (results.caar * 100).round(2).T.to_string(index=False)))

    plot_results(results, False, "esfr_4_complete.pdf")


if __name__ == '__main__':
    main()
