import unittest
from eventstudy.dataprovider import DataProvider


class TestDataProvider(unittest.TestCase):

    def setUp(self):

        # GIVEN a DataProvider
        self.data_provider = DataProvider()

    def test_get_event_window_columns(self):

        # AND event window periods
        num_pre_event_window_periods = num_post_event_window_periods = 6

        # WHEN asked to get_event_window_columns
        actual = self.data_provider.get_event_window_columns(num_pre_event_window_periods,
                                                             num_post_event_window_periods)

        # THEN
        expected = ['-6', '-5', '-4', '-3', '-2', '-1', '0', '1', '2', '3', '4', '5', '6']
        self.assertListEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()

