import datetime
import logging
from eventstudy.eventstudy import *
from eventstudy.ibdatacache import IBDataCache
from examples.example_event_matrix import ExampleEventMatrix


def main():

    # Logging
    logging.basicConfig(filename='naive_ib.log', filemode='w', level=logging.DEBUG, format='%(message)s')
    logger = logging.getLogger()

    file_log_handler = logging.FileHandler('naive_ib.log')
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
    look_back = 0
    look_forward = 10
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
    car, std_err, num_events = calculator.calculate_using_naive_benchmark(
        event_matrix, stock_data['Close'], market_symbol, look_back, look_forward)

    logger.info("CAR: {}".format(car))

    plotter = Plotter()
    plotter.plot_car(car, std_err, num_events,look_back, look_forward, False, "naive_ib.pdf")


if __name__ == "__main__":
    main()