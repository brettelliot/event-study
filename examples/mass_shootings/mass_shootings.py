import pandas as pd
from eventstudy.naivemodel import EventStudyNaiveModel
from eventstudy.dpyahoo import DataProviderYahoo
import datetime as dt


def read_events(file_name, start_date, end_date, value_threshold=7):

    """Read a csv and return a list of events as a pandas DataFrame."""

    event_list_df = pd.read_csv(file_name,
                                usecols=['day_0_date', 'fatalities', 'ticker'],
                                parse_dates=['day_0_date'],
                                )

    # Add index and sort by date
    event_list_df = event_list_df.set_index('day_0_date')
    event_list_df = event_list_df.sort_index()
    #print(event_list_df)
    #print(event_list_df.loc['2001':'2002'])

    # Select between certain dates.
    event_list_df = event_list_df.loc[start_date:end_date]

    # Drop events that don't meet a certain threshold
    event_list_df = event_list_df[event_list_df['fatalities'] >= value_threshold]
    event_list_df = event_list_df.drop(['fatalities'], axis=1)

    # Reset index so day_0_date is a column again
    event_list_df = event_list_df.reset_index()
    #print(event_list_df)
    #print(event_list_df.loc['2017':'2018'])

    return event_list_df


def main():

    start_date = dt.datetime(2016, 1, 1)
    end_date = dt.datetime(2018, 12, 31)
    value_threshold = 7

    event_list_df = read_events('mass_shootings.csv', start_date, end_date, value_threshold)
    # print('The event list:\n {}'.format(event_list_df))

    data_provider = DataProviderYahoo()

    event_study = EventStudyNaiveModel(data_provider, event_list_df)

    # Run the event study looking 6 periods before the event and 6 periods after the event
    num_pre_event_window_periods = num_post_event_window_periods = 6
    market_ticker = 'SPY'
    results = event_study.run_naive_model(market_ticker, num_pre_event_window_periods, num_post_event_window_periods)

    print('\nStarted with {} events and processed {} events.'.format(results.num_starting_events,
                                                                    results.num_events_processed))

    print('\nAAR (%) for all the securities over the event window:\n{}'.format(
        (results.aar * 100).round(2).to_frame().T.to_string(index=False)))

    print('\nCAAR (%) for all the securities over the event window:\n{}'.format(
        (results.caar * 100).round(2).to_frame().T.to_string(index=False)))

    results.plot("Mass shootings and their impact on stock returns", False, 'mass_shootings.pdf')


if __name__ == '__main__':
    main()
