import datetime
import logging
from eventstudy.eventstudy import *
from eventstudy.ibdatacache import IBDataCache
from examples.example_event_matrix import ExampleEventMatrix


def main():

    # Logging
    logging.basicConfig(filename='single_factor_ib.log', filemode='w', level=logging.DEBUG, format='%(message)s')
    logger = logging.getLogger()

    file_log_handler = logging.FileHandler('single_factor_ib.log')
    logger.addHandler(file_log_handler)

    stderr_log_handler = logging.StreamHandler()
    logger.addHandler(stderr_log_handler)
    logger.setLevel('DEBUG')

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
    logger.info("Collecting historical stock data")
    keys = ['Close', 'Volume']
    data_cache = IBDataCache(data_path='../data/ib/')
    stock_data = data_cache.get_multiple_years_of_daily_bars_for_multiple_stocks_as_midf(
        symbols, keys, start_date.year, end_date.year, 0)

    logger.info("Building event matrix")
    eem = ExampleEventMatrix(stock_data.index.levels[1], symbols,
                             value_threshold, csv_file_name)

    # Get a dataframe with an index of all trading days, and columns of all symbols.
    event_matrix = eem.build_event_matrix(start_date, end_date)
    logger.info(event_matrix[(event_matrix == 1.0).any(axis=1)])
    logger.info("Number of events:" + str(len(event_matrix[(event_matrix == 1.0).any(axis=1)])))

    calculator = Calculator()
    ccr = calculator.calculate_using_single_factor_benchmark(event_matrix, stock_data,
                                                             market_symbol, estimation_window, buffer,
                                                             pre_event_window, post_event_window)

    # print results to file and Plots
    logger.info('CARS and CAVCS results for the whole group of stocks')
    logger.info('  Number of events  =  ' + str(ccr.num_events))
    logger.info('CARS Results')
    logger.info('  Number of stocks with +CARS = ' + str(ccr.cars_num_stocks_positive))
    logger.info('  Number of stocks with -CARS = ' + str(ccr.cars_num_stocks_negative))
    logger.info('  CARS t-test value = ' + str(ccr.cars_t_test))
    logger.info('  CARS significant = ' + str(ccr.cars_significant))
    logger.info('  CARS positive = ' + str(ccr.cars_positive))
    logger.info('CAVCS Results')
    logger.info('  Number of stocks with +CAVCS = ' + str(ccr.cavcs_num_stocks_positive))
    logger.info('  Number of stocks with -CAVCS = ' + str(ccr.cavcs_num_stocks_negative))
    logger.info('  CAVCS full t-test value = ' + str(ccr.cavcs_t_test))
    logger.info('  CAVCS significant = ' + str(ccr.cavcs_significant))
    logger.info('  CAVCS positive = ' + str(ccr.cavcs_positive))

    logger.info("CAR: {}".format(ccr.cars))

    plotter = Plotter()
    plotter.plot_car_cavcs(ccr.num_events, ccr.cars, ccr.cars_std_err, ccr.cavcs, ccr.cavcs_std_err,
                           pre_event_window, post_event_window, False, 'single_factor_ib.pdf')


if __name__ == "__main__":
    main()
