import unittest
import datetime as dt
from eventstudy.eventstudy import *
from eventstudy.ibdatareader import IBDataReader
from tests.test_eventstudy.testcarseventstudy import TestCARSEventMatrix

class TestSingleFactorIBEventStudy(unittest.TestCase):

    def setUp(self):

        # Define the symbols to study
        #self.symbols = ['AES', 'AET', 'AFL', 'AVP', 'CLX', 'GM']
        self.symbols = ['AES']

        # Define the market symbol to compare against
        self.market_symbol = "SPY"

        # Add the market symbol to the symbols list to get it's data too
        self.symbols.append(self.market_symbol)

        # Define the start and end date of the study
        self.start_date = dt.datetime(2012, 1, 1)
        self.end_date = dt.datetime(2015, 12, 31)

        # Get a pandas multi-indexed dataframe indexed by date and symbol
        keys = ['Close', 'Volume']
        data_reader = IBDataReader(data_path='../test_data/ib')
        self.stock_data = data_reader.get_multiple_years_of_daily_bars_for_multiple_stocks_as_midf(
            self.symbols, keys, self.start_date.year, self.end_date.year, 1)

        self.symbols.remove('SPY')

        #print(stock_data.loc['AES'])
        #print(stock_data.loc['AES']['Close'])
        #print(stock_data.loc['AES'][stock_data.loc['RGR'].index=='2013-12-30'])

        self.stock_data.drop_duplicates(inplace=True)

    #@unittest.skip("skip")
    def test_posivite_cars(self):
        daily_diff = 0.01
        estimation_window = 200
        buffer = 5
        pre_event_window = 0
        post_event_window = 5
        positive = True

        em = TestCARSEventMatrix(self.stock_data.index.levels[1], self.symbols, self.market_symbol,
                                 self.stock_data, daily_diff, positive)

        # Get a dataframe with an index of all trading days, and columns of all symbols.
        event_matrix = em.build_event_matrix(self.start_date, self.end_date)
        #print(event_matrix[(event_matrix == 1.0).any(axis=1)])

        calculator = Calculator()

        ccr = calculator.calculate_using_single_factor_benchmark(event_matrix, self.stock_data,
                                                                 self.market_symbol, estimation_window, buffer,
                                                                 pre_event_window, post_event_window)

        print(ccr.results_as_string())
        self.assertTrue(ccr.cars_significant)
        self.assertTrue(ccr.cars_positive)

        #plotter = Plotter()

        #plotter.plot_car_cavcs(ccr.num_events, ccr.cars, ccr.cars_std_err, ccr.cavcs, ccr.cavcs_std_err,
        #                       pre_event_window, post_event_window, True)

    def test_negative_cars(self):
        daily_diff = -0.03
        estimation_window = 200
        buffer = 5
        pre_event_window = 0
        post_event_window = 5
        positive = False

        em = TestCARSEventMatrix(self.stock_data.index.levels[1], self.symbols, self.market_symbol,
                                 self.stock_data, daily_diff, positive)

        # Get a dataframe with an index of all trading days, and columns of all symbols.
        event_matrix = em.build_event_matrix(self.start_date, self.end_date)
        #print(event_matrix[(event_matrix == 1.0).any(axis=1)])

        calculator = Calculator()

        ccr = calculator.calculate_using_single_factor_benchmark(event_matrix, self.stock_data,
                                                                 self.market_symbol, estimation_window, buffer,
                                                                 pre_event_window, post_event_window)

        print(ccr.results_as_string())
        self.assertTrue(ccr.cars_significant)
        self.assertFalse(ccr.cars_positive)

        #plotter = Plotter()

        #plotter.plot_car_cavcs(ccr.num_events, ccr.cars, ccr.cars_std_err, ccr.cavcs, ccr.cavcs_std_err,
        #                       pre_event_window, post_event_window, True)


if __name__ == '__main__':
    unittest.main()
