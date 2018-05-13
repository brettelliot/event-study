import datetime
import logging
from eventstudy.eventstudy import *
from eventstudy.ibdatacache import IBDataCache
from examples.example_event_matrix import ExampleEventMatrix

import warnings
warnings.filterwarnings("ignore")


def main():

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        #level=logging.DEBUG,
        format='%(message)s',
        handlers=[
            logging.FileHandler('naive_ib.log', mode='w'),
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
    value_threshold = 10
    look_back = 0
    look_forward = 10
    csv_file_name = '../data/events/event_dates.csv'

    # Get a pandas multi-indexed dataframe indexed by date and symbol
    logger.debug("Collecting historical stock data")
    keys = ['Close', 'Volume']
    data_cache = IBDataCache(data_path='../data/ib/')
    stock_data = data_cache.get_multiple_years_of_daily_bars_for_multiple_stocks_as_midf(
        symbols, keys, start_date.year, end_date.year, 0)
        
        
    #stock_data.loc[('RGR',slice(None))]


    #print(stock_data.loc['RGR'][stock_data.loc['RGR'].index=='2011-12-30'])
    
    '''
        Open   High    Low  Close  Volume  OpenInterest
        Date
        2011-12-30  33.82  34.05  33.45  33.46     709             0
        2011-12-30  33.82  34.05  33.45  33.46     709             0  '''
    
    stock_data.drop_duplicates(inplace = True)
    
    #print(stock_data.loc['RGR'][stock_data.loc['RGR'].index=='2011-12-30'])
    
    #2011-12-30  33.82  34.05  33.45  33.46     709             0


    logger.debug("Building event matrix")
    eem = ExampleEventMatrix(stock_data.index.levels[1], symbols,
                             value_threshold, csv_file_name)

    # Get a dataframe with an index of all trading days, and columns of all symbols.
    event_matrix = eem.build_event_matrix(start_date, end_date)
    logger.debug(event_matrix[(event_matrix == 1.0).any(axis=1)])
    logger.info("Number of events:" + str(len(event_matrix[(event_matrix == 1.0).any(axis=1)])))

    calculator = Calculator()

    ccr = calculator.calculate_using_naive_benchmark(
        event_matrix, stock_data['Close'], market_symbol, look_back, look_forward)

    logger.info(ccr.results_as_string())

    plotter = Plotter()

    plotter.plot_car(ccr.cars, ccr.cars_std_err, ccr.num_events,look_back, look_forward, False, "naive_ib.pdf")


if __name__ == "__main__":
    main()
