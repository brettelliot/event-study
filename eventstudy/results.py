

class EventStudyResults(object):
    """Helper class that collects and formats Event Study Results."""

    def __init__(self):
        self.num_starting_events = 0
        self.num_events_processed = 0
        self.aar = None
        self.caar = None
        self.std_err = None
