import datetime
import logging
from eventstudy.eventstudy import *
from eventstudy.avdatacache import AVDataCache
from examples.example_event_matrix import ExampleEventMatrix


def main():

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        #level=logging.DEBUG,
        format='%(message)s',
        handlers=[
            logging.FileHandler('single_factor_av.log', mode='w'),
            logging.StreamHandler()
        ])
    logger = logging.getLogger()

    # Define the symbols to study
    symbols = ['RGR', 'OLN']

    # Define the market symbol to compare against
    market_symbol = "SPY"

    # Add the market symbol to the symbols list to get it's data too
    symbols.append(market_symbol)

    # Define the start and end date of the study
    start_date = datetime.datetime(2000, 1, 1)
    end_date = datetime.datetime(2017, 12, 31)
    value_threshold = 1
    estimation_window = 200
    buffer = 5
    pre_event_window = 0
    post_event_window = 10
    csv_file_name = '../data/events/event_dates.csv'

    # Get a pandas multi-indexed dataframe indexed by date and symbol
    logger.debug("Collecting historical stock data")
    keys = ['adjusted_close', 'volume']
    stock_data_cache = AVDataCache('../data/av/')
    stock_data = stock_data_cache.get_stock_data(symbols, keys)

    logger.debug("Building event matrix")
    eem = ExampleEventMatrix(stock_data.index.levels[1], symbols,
                             value_threshold, csv_file_name)

    # Get a dataframe with an index of all trading days, and columns of all symbols.
    event_matrix = eem.build_event_matrix(start_date, end_date)
    logger.debug(event_matrix[(event_matrix == 1.0).any(axis=1)])
    logger.info("Number of events:" + str(len(event_matrix[(event_matrix == 1.0).any(axis=1)])))

    calculator = Calculator()
    ccr = calculator.calculate_using_single_factor_benchmark(event_matrix, stock_data,
                                                             market_symbol, estimation_window, buffer,
                                                             pre_event_window, post_event_window)

    logger.info(ccr.results_as_string())

    plotter = Plotter()
    plotter.plot_car_cavcs(ccr.num_events, ccr.cars, ccr.cars_std_err, ccr.cavcs, ccr.cavcs_std_err,
                           pre_event_window, post_event_window, False, 'single_factor_av.pdf')


if __name__ == "__main__":
    main()
