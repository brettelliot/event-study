import pandas as pd
from eventstudy.naivemodel import EventStudyNaiveModel
from eventstudy.dpyahoo import DataProviderYahoo


def read_events(file_name):

    """Read a csv and return a list of events as a pandas DataFrame."""

    event_list_df = pd.read_csv(file_name,
                                usecols=['day_0_date', 'ticker'],
                                parse_dates=['day_0_date'],
                                )

    return event_list_df


def main():

    """
    This example shows the impact of severe airline crashes and their impact on stock returns.
    """

    event_list_df = read_events('airline_crashes.csv')
    # print('The event list:\n {}'.format(event_list_df))

    data_provider = DataProviderYahoo()

    event_study = EventStudyNaiveModel(data_provider, event_list_df)

    # Run the event study looking 6 periods before the event and 6 periods after the event
    num_pre_event_window_periods = num_post_event_window_periods = 6
    market_ticker = 'SPY'
    results = event_study.run_naive_model(market_ticker, num_pre_event_window_periods, num_post_event_window_periods)

    print('\nStarted with {} events and processed {} events.'.format(results.num_starting_events,
                                                                    results.num_events_processed))

    results.plot("Airline Crashes and their impact on stock returns", False, 'airline_crashes.pdf')

    print('\nCAAR (%) for all the securities over the event window:\n{}'.format(
        (results.caar * 100).round(2).to_frame().T.to_string(index=False)))

    print('\nAAR (%) for all the securities over the event window:\n{}'.format(
        (results.aar * 100).round(2).to_frame().T.to_string(index=False)))


if __name__ == '__main__':
    main()
