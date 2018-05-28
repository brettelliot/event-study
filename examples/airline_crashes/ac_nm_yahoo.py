import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from eventstudy.naivemodel import EventStudyNaiveModel
from eventstudy.dpyahoo import DataProviderYahoo


def read_events(file_name):

    """Read a csv and return a list of events as a pandas DataFrame."""

    event_list_df = pd.read_csv(file_name,
                                usecols=['day_0_date', 'ticker'],
                                parse_dates=['day_0_date'],
                                )

    return event_list_df


def plot_results(results, show=True, pdf_filename=None):
    plt.clf()
    plt.figure(1)
    ymin = min(np.nanmin(results.caar), np.nanmin(results.aar)) * 100 - .5
    ymax = max(np.nanmax(results.caar), np.nanmax(results.aar)) * 100 + .5

    ax1 = plt.subplot(211)
    plt.grid()
    ax1.set_ylim([ymin, ymax])
    plt.plot(results.caar * 100, label="N=%s" % results.num_events_processed)
    plt.legend(loc='upper right')
    plt.title("CAAR before and after event")
    plt.ylabel('CAAR (%)')
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.xlabel('Event Window')

    ax2 = plt.subplot(212)
    plt.grid()
    ax2.set_ylim([ymin, ymax])
    plt.plot(results.aar * 100, label="N=%s" % results.num_events_processed)
    plt.legend(loc='upper right')
    plt.title("AAR before and after event")
    plt.ylabel('AAR (%)')
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.xlabel('Event Window')

    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

    if pdf_filename is not None:
        plt.savefig(pdf_filename, format='pdf')
    if show:
        plt.show()


def main():

    """
    This example replicates "Event Studies for Financial Research, chapter 4:
    A simplified example, the effect of air crashes on stock prices"
    using the eventstudy package.
    """

    event_list_df = read_events('ac_events.csv')
    # print('The event list:\n {}'.format(event_list_df))

    data_provider = DataProviderYahoo()

    event_study = EventStudyNaiveModel(data_provider, event_list_df)

    # Run the event study looking 6 periods before the event and 6 periods after the event
    num_pre_event_window_periods = num_post_event_window_periods = 6
    market_ticker = 'SPY'
    results = event_study.run_naive_model(market_ticker, num_pre_event_window_periods, num_post_event_window_periods)

    print('\nCAAR (%) for all the securities over the event window:\n{}'.format((results.caar * 100).round(2).to_frame().T.to_string(index=False)))

    plot_results(results, False, 'ac_nm_yahoo.pdf')

    print('Started with {} events and processed {} events'.format(results.num_starting_events, results.num_events_processed))


if __name__ == '__main__':
    main()
