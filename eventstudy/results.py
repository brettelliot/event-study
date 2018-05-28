import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

class EventStudyResults(object):
    """Helper class that collects and formats Event Study Results."""

    def __init__(self):
        self.num_starting_events = 0
        self.num_events_processed = 0
        self.aar = None
        self.caar = None
        self.std_err = None

    def plot(self, title=None, show=True, pdf_filename=None, ):
        plt.clf()
        plt.figure(1)

        # Use the same bounds for all charts
        #ymin = min(np.nanmin(self.caar), np.nanmin(self.aar)) * 100 - .5
        #ymax = max(np.nanmax(self.caar), np.nanmax(self.aar)) * 100 + .5

        ax1 = plt.subplot(211)
        plt.title(title)
        plt.grid()
        ymin = np.nanmin(self.caar) * 100 - .5
        ymax = np.nanmax(self.caar) * 100 + .5
        ax1.set_ylim([ymin, ymax])
        plt.ylabel('CAAR (%)')
        ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.plot(self.caar * 100, label="N=%s" % self.num_events_processed)
        plt.legend(loc='upper right')
        #plt.xlabel('Event Window')

        ax2 = plt.subplot(212)
        plt.grid()
        ymin = np.nanmin(self.aar) * 100 - .5
        ymax = np.nanmax(self.aar) * 100 + .5
        ax2.set_ylim([ymin, ymax])
        plt.plot(self.aar * 100, label="N=%s" % self.num_events_processed)
        plt.legend(loc='upper right')
        #plt.title("AAR before and after event")
        plt.ylabel('AAR (%)')
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.xlabel('Event Window')

        plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

        if pdf_filename is not None:
            plt.savefig(pdf_filename, format='pdf')
        if show:
            plt.show()
