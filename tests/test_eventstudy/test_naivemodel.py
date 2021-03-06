import unittest
import numpy as np
import pandas as pd
from io import StringIO
from eventstudy.naivemodel import EventStudyNaiveModel
from eventstudy.dataprovider import DataProvider
from pandas.util.testing import assert_frame_equal
from pandas.util.testing import assert_series_equal


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


class DataProviderESFR(DataProvider):
    """Provide security price data specifically for this event study."""

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


class TestEventStudyNaiveModel(unittest.TestCase):

    def setUp(self):

        # GIVEN an event list of airline crashes
        event_list_df = load_events()

        # AND security price data around the crash date for airlines involved
        data_provider = DataProviderESFR()

        # AND an Event Study using the naive model
        event_study = EventStudyNaiveModel(data_provider, event_list_df)

        # AND the parameters needed to run the model
        num_pre_event_window_periods = num_post_event_window_periods = 6
        market_ticker = 'SPY'


        # WHEN we run the naive model
        self.results = event_study.run_naive_model(market_ticker,
                                                   num_pre_event_window_periods,
                                                   num_post_event_window_periods)



    def test_aar(self):

        # THEN we should get the expected CAARs
        columns = ['-6', '-5', '-4', '-3', '-2', '-1', '0', '1', '2', '3', '4', '5', '6']
        aar = [np.nan, -0.000295, 0.001173, 0.005690, 0.005685, -0.015915, -0.029484, -0.000439, 0.004728,
               0.002269, 0.014150, 0.003878, -0.004316]
        expected = pd.Series(data=aar, index=columns, dtype='float64')
        #print((self.results.aar * 100).round(2).to_frame().T)
        assert_series_equal(expected.round(6), self.results.aar.round(6).T)


    def test_caar(self):

        # THEN we should get the expected CAARs
        columns = ['-6', '-5', '-4', '-3', '-2', '-1', '0', '1', '2', '3', '4', '5', '6']
        caar = [np.nan, -0.000295, 0.000878, 0.006568, 0.012253, -0.003662, -0.033146, -0.033585, -0.028856,
                -0.026587, -0.012438, -0.008559, -0.012875]
        expected = pd.Series(data=caar, index=columns, dtype='float64')
        assert_series_equal(expected.round(6), self.results.caar.round(6).T)


if __name__ == '__main__':
    unittest.main()
