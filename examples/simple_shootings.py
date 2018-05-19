
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
import datetime as dt

from eventstudy.ibdatacache import IBDataCache

"""
Get just the closing and open prices for our symbols
"""

"""
symbols = ['LUV', 'DAL', 'AAL', 'ALK', 'LCC', 'SPY']

event_start_date = dt.datetime(1999, 1, 1)
event_end_date = dt.datetime(2018, 12, 31)


keys = ['Close', 'Volume']
data_reader = IBDataReader(data_path='../data/ib/')
data = data_reader.get_multiple_years_of_daily_bars_for_multiple_stocks_as_midf(
    symbols, keys, event_start_date.year, event_end_date.year, 1)

market_symbol = 'SPY'
symbols.remove(market_symbol)

#print(data.loc['LUV']['Close'])


events = [
    #    {'Date': dt.datetime.strptime("1999-06-01", "%Y-%m-%d"), 'value': 11, 'symbol':'AAL'},
    #    {'Date': dt.datetime.strptime("2000-01-31", "%Y-%m-%d"), 'value': 88, 'symbol':'ALK'},
    #    {'Date': dt.datetime.strptime("2000-03-05", "%Y-%m-%d"), 'value': 0, 'symbol':'LUV'},
    #    {'Date': dt.datetime.strptime("2001-11-12", "%Y-%m-%d"), 'value': 260, 'symbol':'AAL'},
    {'Date': dt.datetime.strptime("2005-12-08", "%Y-%m-%d"), 'value': 0, 'symbol':'LUV'},
    {'Date': dt.datetime.strptime("2009-01-15", "%Y-%m-%d"), 'value': 0, 'symbol':'LCC'},
    {'Date': dt.datetime.strptime("2009-07-13", "%Y-%m-%d"), 'value': 0, 'symbol':'LUV'},
    {'Date': dt.datetime.strptime("2009-12-22", "%Y-%m-%d"), 'value': 0, 'symbol':'AAL'},
    {'Date': dt.datetime.strptime("2011-04-01", "%Y-%m-%d"), 'value': 0, 'symbol':'LUV'},
    {'Date': dt.datetime.strptime("2013-07-22", "%Y-%m-%d"), 'value': 0, 'symbol':'lUV'},
    {'Date': dt.datetime.strptime("2015-03-05", "%Y-%m-%d"), 'value': 0, 'symbol':'DAL'},
    {'Date': dt.datetime.strptime("2016-09-27", "%Y-%m-%d"), 'value': 0, 'symbol':'LUV'},
    {'Date': dt.datetime.strptime("2016-10-28", "%Y-%m-%d"), 'value': 20, 'symbol':'AAL'},
    {'Date': dt.datetime.strptime("2018-04-17", "%Y-%m-%d"), 'value': 1, 'symbol':'LUV'}
]

"""

symbols = ['RGR', 'OLN', 'SPY']

event_start_date = dt.datetime(2000, 1, 1)
event_end_date = dt.datetime(2017, 12, 31)

keys = ['Close', 'Volume']
data_cache = IBDataCache(data_path='../data/ib/')
data = data_cache.get_multiple_years_of_daily_bars_for_multiple_stocks_as_midf(
    symbols, keys, event_start_date.year, event_end_date.year, 0)

market_symbol = 'SPY'
symbols.remove(market_symbol)

#print(data.index.levels)
#print(data.loc['LUV']


# Hard code some event data to test with
events = [
    {'Date': dt.datetime.strptime("2009-03-30", "%Y-%m-%d"), 'symbol': 'RGR'},
    {'Date': dt.datetime.strptime("2009-04-03", "%Y-%m-%d"), 'symbol': 'RGR'},
    {'Date': dt.datetime.strptime("2009-11-05", "%Y-%m-%d"), 'symbol': 'RGR'},
    {'Date': dt.datetime.strptime("2010-08-03", "%Y-%m-%d"), 'symbol': 'RGR'},
    {'Date': dt.datetime.strptime("2011-10-14", "%Y-%m-%d"), 'symbol': 'RGR'},
    {'Date': dt.datetime.strptime("2012-04-02", "%Y-%m-%d"), 'symbol': 'RGR'},
    {'Date': dt.datetime.strptime("2012-07-20", "%Y-%m-%d"), 'symbol': 'RGR'},
    {'Date': dt.datetime.strptime("2012-08-06", "%Y-%m-%d"), 'symbol': 'RGR'},
    {'Date': dt.datetime.strptime("2012-09-27", "%Y-%m-%d"), 'symbol': 'RGR'},
    {'Date': dt.datetime.strptime("2012-12-14", "%Y-%m-%d"), 'symbol': 'RGR'},
    {'Date': dt.datetime.strptime("2013-07-26", "%Y-%m-%d"), 'symbol': 'RGR'},
    {'Date': dt.datetime.strptime("2013-09-16", "%Y-%m-%d"), 'symbol': 'RGR'},
    {'Date': dt.datetime.strptime("2015-06-17", "%Y-%m-%d"), 'symbol': 'RGR'},
    {'Date': dt.datetime.strptime("2015-10-01", "%Y-%m-%d"), 'symbol': 'RGR'},
    {'Date': dt.datetime.strptime("2015-12-02", "%Y-%m-%d"), 'symbol': 'RGR'},
    {'Date': dt.datetime.strptime("2016-06-13", "%Y-%m-%d"), 'symbol': 'RGR'},

    {'Date': dt.datetime.strptime("2009-03-30", "%Y-%m-%d"), 'symbol': 'OLN'},
    {'Date': dt.datetime.strptime("2009-04-03", "%Y-%m-%d"), 'symbol': 'OLN'},
    {'Date': dt.datetime.strptime("2009-11-05", "%Y-%m-%d"), 'symbol': 'OLN'},
    {'Date': dt.datetime.strptime("2010-08-03", "%Y-%m-%d"), 'symbol': 'OLN'},
    {'Date': dt.datetime.strptime("2011-10-14", "%Y-%m-%d"), 'symbol': 'OLN'},
    {'Date': dt.datetime.strptime("2012-04-02", "%Y-%m-%d"), 'symbol': 'OLN'},
    {'Date': dt.datetime.strptime("2012-07-20", "%Y-%m-%d"), 'symbol': 'OLN'},
    {'Date': dt.datetime.strptime("2012-08-06", "%Y-%m-%d"), 'symbol': 'OLN'},
    {'Date': dt.datetime.strptime("2012-09-27", "%Y-%m-%d"), 'symbol': 'OLN'},
    {'Date': dt.datetime.strptime("2012-12-14", "%Y-%m-%d"), 'symbol': 'OLN'},
    {'Date': dt.datetime.strptime("2013-07-26", "%Y-%m-%d"), 'symbol': 'OLN'},
    {'Date': dt.datetime.strptime("2013-09-16", "%Y-%m-%d"), 'symbol': 'OLN'},
    {'Date': dt.datetime.strptime("2015-06-17", "%Y-%m-%d"), 'symbol': 'OLN'},
    {'Date': dt.datetime.strptime("2015-10-01", "%Y-%m-%d"), 'symbol': 'OLN'},
    {'Date': dt.datetime.strptime("2015-12-02", "%Y-%m-%d"), 'symbol': 'OLN'},
    {'Date': dt.datetime.strptime("2016-06-13", "%Y-%m-%d"), 'symbol': 'OLN'}
]

event_dates_df = pd.DataFrame(events)
event_dates_df = event_dates_df.set_index('Date')
#print('event_dates_df:')

#print(event_dates_df)

"""
Step Two: Creating some helper functions to find our open prices and close prices
"""


def get_close_price(data, symbol, current_date, day_number):
    #: If we're looking at day 0 just return the indexed date
    if day_number == 0:
        return data.loc[symbol]['Close'].ix[current_date]
    #: Find the close price day_number away from the current_date
    else:
        #: If the close price is too far ahead, just get the last available
        total_date_index_length = len(data.loc[symbol]['Close'].index)
        #: Find the closest date to the target date
        date_index = data.loc[symbol]['Close'].index.searchsorted(current_date + dt.timedelta(day_number))
        #: If the closest date is too far ahead, reset to the latest date possible
        date_index = total_date_index_length - 1 if date_index >= total_date_index_length else date_index
        #: Use the index to return a close price that matches
        return data.loc[symbol].iloc[date_index]


def get_first_price(data, starting_point, symbol, date):
    starting_day = date - dt.timedelta(starting_point)
    date_index = data.loc[symbol]['Close'].index.searchsorted(starting_day)
    # date_index = data.loc[symbol]['Close'].index.searchsorted(starting_day, side='left')
    return data.loc[symbol].iloc[date_index]


def remove_outliers(returns, num_std_devs):
    return returns[~((returns - returns.mean()).abs() > num_std_devs * returns.std())]


def get_returns(data, starting_point, symbol, date, day_num):
    #: Get stock prices
    first_price = get_first_price(data, starting_point, symbol, date)
    close_price = get_close_price(data, symbol, date, day_num)

    #: Calculate returns
    ret = (close_price - first_price) / (first_price + 0.0)
    return ret


"""
Step Three: Calculate average cumulative returns
"""

#: Dictionaries that I'm going to be storing calculated data in
all_returns = {}
all_std_devs = {}
total_sample_size = {}

#: Create our range of day_numbers that will be used to calculate returns
starting_point = 30
#: Looking from -starting_point till +starting_point which creates our timeframe band
day_numbers = [i for i in range(-starting_point, starting_point)]

for day_num in day_numbers:

    #: Reset our returns and sample size each iteration
    returns = []
    sample_size = 0

    #: Get the return compared to t=0
    for date, row in event_dates_df.iterrows():
        symbol = row.symbol

        #: Make sure that data exists for the dates
        if date not in data.index.levels[1] or symbol not in data.index.levels[0]:
            continue

        returns.append(get_returns(data, starting_point, symbol, date, day_num))
        sample_size += 1

    #: Drop any Nans, remove outliers, find outliers and aggregate returns and std dev
    returns = pd.Series(returns).dropna()
    returns = remove_outliers(returns, 2)
    all_returns[day_num] = np.average(returns)
    all_std_devs[day_num] = np.std(returns)
    total_sample_size[day_num] = sample_size

#: Take all the returns, stds, and sample sizes that I got and put that into a Series
all_returns = pd.Series(all_returns)
all_std_devs = pd.Series(all_std_devs)
N = np.average(pd.Series(total_sample_size))

print(all_returns)

"""
Step Four: Plotting our event study graph
"""

xticks = [d for d in day_numbers if d % 2 == 0]
all_returns.plot(xticks=xticks, label="N=%s" % N)

pyplot.grid(b=None, which=u'major', axis=u'y')
pyplot.title("Cumulative Return before and after event")
pyplot.xlabel("Window Length (t)")
pyplot.legend()
pyplot.ylabel("Cumulative Return (r)")
#pyplot.show()
