import datetime
import logging
from eventstudy.eventstudy import *
from eventstudy.ibdatareader import IBDataReader
from examples.airline_crashes.ac_event_matrix import AirlineCrashEventMatrix


def main():

    # Logging
    logging.basicConfig(
        #level=logging.INFO,
        level=logging.DEBUG,
        format='%(message)s',
        handlers=[
            logging.FileHandler('ac_single_factor_ib.log', mode='w'),
            logging.StreamHandler()
        ])
    logger = logging.getLogger()

    # Define the symbols to study
    #symbols = ['LUV', 'DAL', 'AAL', 'ALK']
    symbols = ['LUV', 'ALK']

    # Define the market symbol to compare against
    market_symbol = "SPY"

    # Add the market symbol to the symbols list to get it's data too
    symbols.append(market_symbol)

    # Define the start and end date of the study
    start_date = datetime.datetime(2001, 1, 1)
    end_date = datetime.datetime(2018, 12, 31)
    value_threshold = 1
    estimation_window = 50
    buffer = 5
    pre_event_window = 0
    post_event_window = 10
    csv_file_name = '../../data/events/airline_crash_event_dates.csv'

    # Get a pandas multi-indexed dataframe indexed by date and symbol
    logger.debug("Collecting historical stock data")
    keys = ['Close', 'Volume']
    data_reader = IBDataReader(data_path='../../data/ib/')
    stock_data = data_reader.get_multiple_years_of_daily_bars_for_multiple_stocks_as_midf(
        symbols, keys, start_date.year, end_date.year, 1)

    logger.debug("Building event matrix")
    em = AirlineCrashEventMatrix(stock_data.index.levels[1], symbols,
                             value_threshold, csv_file_name)

    # Get a dataframe with an index of all trading days, and columns of all symbols.
    event_matrix = em.build_event_matrix(start_date, end_date)
    logger.debug(event_matrix[(event_matrix == 1.0).any(axis=1)])
    logger.info("Number of events: " + str(len(event_matrix[(event_matrix == 1.0).any(axis=1)])))

    calculator = Calculator()
    ccr = calculator.calculate_using_single_factor_benchmark(event_matrix, stock_data,
                                                             market_symbol, estimation_window, buffer,
                                                             pre_event_window, post_event_window)

    logger.info(ccr.results_as_string())

    plotter = Plotter()
    plotter.plot_car_cavcs(ccr.num_events, ccr.cars, ccr.cars_std_err, ccr.cavcs, ccr.cavcs_std_err,
                           pre_event_window, post_event_window, False, 'ac_single_factor_ib.pdf')
    #plotter.plot_car(ccr.cars, ccr.cars_std_err, ccr.num_events, pre_event_window, post_event_window,
    #                 False, "ac_single_factor_ib.pdf")


if __name__ == "__main__":
    main()
