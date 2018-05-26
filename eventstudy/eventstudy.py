import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt



class EventMatrix(object):

    def __init__(self, datetimes, symbols):
        '''
        :param datetimes:
        :param symbols:
        Constructs A pandas dataframe indexed by datetimes and with columns for each symbol.
        The constructor fills this with all NANs and an abstract base method exists to be customized.
        '''

        # Build an empty event matrix with an index of all the datetimes and columns for each symbol.
        # Fill with NANs
        self.event_matrix = pd.DataFrame({'Date': datetimes})
        self.event_matrix = self.event_matrix.set_index('Date')
        self.event_matrix.tz_localize(tz='America/New_York')
        self.event_matrix = self.event_matrix.sort_index()
        self.event_matrix = self.event_matrix.loc[~self.event_matrix.index.duplicated(keep='first')]
        # Prints True is sorted
        #print(self.event_matrix.index.is_monotonic)

        self.symbols = symbols
        for symbol in self.symbols:
            self.event_matrix[symbol] = np.nan


    def build_event_matrix(self, start_date, end_date):
        '''
        Implement this method a derived class.
        :param start_date:
        :param end_date:
        :return: FIll up the event matrix with 1's in the row/column for which there was an event.
        '''
        raise NotImplementedError("Please Implement this method in a base class")


class CarsCavcsResult(object):

    def __init__(self, num_events,
                 cars, cars_std_err, cars_t_test, cars_significant, cars_positive, cars_num_stocks_positive,
                 cars_num_stocks_negative,
                 cavcs, cavcs_std_err, cavcs_t_test, cavcs_significant, cavcs_positive, cavcs_num_stocks_positive,
                 cavcs_num_stocks_negative):
        """
        :param num_events: the number of events in the matrix
        :param cars: time series of Cumulative Abnormal Return
        :params cars_std_err: std error of the CARs
        :param cars_t_test: t-test statistic that checks whether the CARs of all stock are significantly different from 0
        :param cars_significant: True if the CARs of all stocks are significant
        :param cars_positive: True if the CAR is positive
        :param cars_num_stocks_positive: The number of stocks for which the CAR was significantly positive
        :param cars_num_stocks_negative: The number of stocks for which the CAR was significantly negative
        :param cavcs: time series of Cumulative Abnormal Volume Changes
        :params cavcs_std_err: std error of the CAVCs
        :param cavcs_t_test: t-test statistic that checks whether the CAVCs of all stock are significantly different from 0
        :param cavcs_significant: True if the CAVCs of all stocks are significant
        :param cavcs_positive: True if the CAVC is  positive
        :param cavcs_num_stocks_positive: The number of stocks for which the CAVC was significantly positive
        :param cavcs_num_stocks_negative: The number of stocks for which the CAVC was significantly negative

        All of the above t-tests are significant when they are in the 95% confidence levels
        """
        self.num_events = num_events
        self.cars = cars
        self.cars_std_err = cars_std_err
        self.cars_t_test = cars_t_test
        self.cars_significant = cars_significant
        self.cars_positive = cars_positive
        self.cars_num_stocks_positive = cars_num_stocks_positive
        self.cars_num_stocks_negative = cars_num_stocks_negative
        self.cavcs = cavcs
        self.cavcs_std_err = cavcs_std_err
        self.cavcs_t_test = cavcs_t_test
        self.cavcs_significant = cavcs_significant
        self.cavcs_positive = cavcs_positive
        self.cavcs_num_stocks_positive = cavcs_num_stocks_positive
        self.cavcs_num_stocks_negative = cavcs_num_stocks_negative

    def results_as_string(self):
        result_string = 'Number of events processed: ' + str(self.num_events) + '\n'
        result_string += 'CARS Results' + '\n'
        result_string += '  Number of stocks with +CARS: ' + str(self.cars_num_stocks_positive) + '\n'
        result_string += '  Number of stocks with -CARS: ' + str(self.cars_num_stocks_negative) + '\n'
        result_string += '  CARS t-test value: ' + str(self.cars_t_test) + '\n'
        result_string += '  CARS significant: ' + str(self.cars_significant) + '\n'
        result_string += '  CARS positive: ' + str(self.cars_positive) + '\n'
        result_string += 'CAVCS Results' + '\n'
        result_string += '  Number of stocks with +CAVCS: ' + str(self.cavcs_num_stocks_positive) + '\n'
        result_string += '  Number of stocks with -CAVCS: ' + str(self.cavcs_num_stocks_negative) + '\n'
        result_string += '  CAVCS full t-test value: ' + str(self.cavcs_t_test) + '\n'
        result_string += '  CAVCS significant: ' + str(self.cavcs_significant) + '\n'
        result_string += '  CAVCS positive: ' + str(self.cavcs_positive) + '\n'

        return result_string


class Calculator(object):
    def __init__(self):
        pass

    def calculate_using_naive_benchmark(self, event_matrix, stock_data, market_symbol, look_back, look_forward):
        """
        :param event_matrix:
        :param stock_data:
        :param market_symbol:
        :param look_back:
        :param look_forward:
        :return car: time series of Cumulative Abnormal Return
        :return std_err: the standard error
        :return num_events: the number of events in the matrix

        Most of the code was from here:
        https://github.com/brettelliot/QuantSoftwareToolkit/blob/master/QSTK/qstkstudy/EventProfiler.py
        """

        # Copy the stock prices into a new dataframe which will become filled with the returns

        #import pdb;
        #pdb.set_trace()

        try:

            # For IB
            daily_returns = stock_data['Close'].copy()
            volumes = stock_data['Volume'].copy()

        except KeyError:

            # For AV
            daily_returns = stock_data['adjusted_close'].copy()
            volumes = stock_data['volume'].copy()

        # Convert prices into daily returns.
        # This is the amount that the specific stock increased or decreased in value for one day.
        daily_returns = daily_returns.pct_change().fillna(0)

        mypct = lambda x: x[-1] - np.mean(x[:-1])

        vlm_changes = volumes.rolling(5, 5).apply(mypct).fillna(0)

        # Subtract the market returns from all of the stock's returns. The result is the abnormal return.
        # beta = get_beta()
        beta = 1.0  # deal with beta later

        symbols = daily_returns.index.get_level_values(0).unique()

        abnormal_returns = daily_returns.copy()
        ex_vols = vlm_changes.copy()


        #import pdb;
        #pdb.set_trace()

        for sym in symbols:
            abnormal_returns.loc[sym, slice(None)] -= beta * daily_returns.loc[market_symbol, slice(None)].values
            ex_vols.loc[sym, slice(None)] -= beta * vlm_changes.loc[market_symbol, slice(None)].values

        #import pdb;
        #pdb.set_trace()

        # remove the market symbol from the returns and event matrix. It's no longer needed.
        del daily_returns[market_symbol]
        del vlm_changes[market_symbol]
        del abnormal_returns[market_symbol]
        del ex_vols[market_symbol]

        try:
            del event_matrix[market_symbol]
        except KeyError as e:
            pass

        starting_event_num = len(event_matrix[(event_matrix == 1.0).any(axis=1)])
        print("Starting number of events: {}".format(starting_event_num))

        # The event matrix has a row for every data in the stock data.
        # Zero (NaN) out any events in the rows at the beginning and end that would
        # not have data.
        event_matrix.values[0:look_back, :] = np.NaN
        event_matrix.values[-look_forward:, :] = np.NaN

        ending_event_num = len(event_matrix[(event_matrix == 1.0).any(axis=1)])
        print("Ending number of events: {}".format(ending_event_num))
        if (starting_event_num != ending_event_num):
            print("{} events were dropped because they require data outside the data range.".format(starting_event_num -
                                                                                                    ending_event_num))
        # Number of events
        i_no_events = int(np.logical_not(np.isnan(event_matrix.values)).sum())
        assert i_no_events > 0, "Zero events in the event matrix"

        na_all_rets = "False"
        na_all_vlms = "False"

        # import pdb; pdb.set_trace()

        results = pd.DataFrame(index=symbols, columns=['pos', 'neg', 'vpos', 'vneg'])

        # Looking for the events and pushing them to a matrix
        #print(event_matrix.columns)
        #print(symbols)
        try:
            #for i, s_sym in enumerate(event_matrix.columns):
            for s_sym in symbols:
                if s_sym == market_symbol:
                    continue
                na_stock_rets = "False"
                na_stock_vlms = "False"
                for j, dt_date in enumerate(event_matrix.index):

                    if event_matrix[s_sym][dt_date] == 1:

                        na_ret = abnormal_returns[s_sym][j - look_back:j + 1 + look_forward]
                        na_vls = ex_vols[s_sym][j - look_back:j + 1 + look_forward]

                        if type(na_stock_rets) == type(""):
                            na_stock_rets = na_ret
                            na_stock_vlms = na_vls
                        else:
                            na_stock_rets = np.vstack((na_stock_rets, na_ret))
                            na_stock_vlms = np.vstack((na_stock_vlms, na_vls))

                # reurns/vols for a particular stock analyze here
                # then append to all rets

                #import pdb;
                #pdb.set_trace()

                if type(na_stock_rets) == type("") or type(na_stock_rets) == type(""):
                    continue

                if (np.mean(na_stock_rets) > 0):
                    results.loc[s_sym, 'pos'] = True
                    #print(s_sym)
                else:
                    results.loc[s_sym, 'neg'] = True

                if (np.mean(na_stock_vlms) > 0):
                    results.loc[s_sym, 'vpos'] = True
                else:
                    results.loc[s_sym, 'vneg'] = True

                if type(na_all_rets) == type(""):
                    na_all_rets = na_stock_rets
                    na_all_vlms = na_stock_vlms
                else:
                    na_all_rets = np.vstack((na_all_rets, na_stock_rets))
                    na_all_vlms = np.vstack((na_all_vlms, na_stock_vlms))


        except Exception as e:
            #import pdb;
            #pdb.set_trace()
            #print(e)
            raise e

        #import pdb;
        #pdb.set_trace()

        if len(na_all_rets.shape) == 1:
            na_all_rets = np.expand_dims(na_all_rets, axis=0)

        # Computing daily returns

        num_events = len(na_all_rets)

        cars = np.mean(na_all_rets, axis=0)
        cavs = np.mean(na_all_vlms, axis=0)

        cars_std_err = np.std(na_all_rets, axis=0)
        cavs_std_err = np.std(na_all_vlms, axis=0)

        na_cum_rets = np.cumprod(na_all_rets + 1, axis=1)
        na_cum_rets = (na_cum_rets.T / na_cum_rets[:, look_back]).T

        na_cum_vlms = np.cumsum(na_all_vlms, axis=1)

        cars_cum = np.mean(na_cum_rets, axis=0)
        cavs_cum = np.mean(na_cum_vlms, axis=0)

        if (cars_cum[-1] - 1) > 0:
            cars_positive = True
        else:
            cars_positive = False

        if (cavs_cum[-1]) > 0:
            cavs_positive = True
        else:
            cavs_positive = False

        cars_num_stocks_positive = results['pos'].sum()

        cars_num_stocks_negative = results['neg'].sum()

        cavs_num_stocks_positive = results['vpos'].sum()

        cavs_num_stocks_negative = results['vneg'].sum()

        std1 = np.std(cars)

        cars_t_test = np.mean(cars) / std1 * np.sqrt(len(cars))

        std2 = np.std(cavs)
        cavs_t_test = np.mean(cavs) / std2 * np.sqrt(len(cavs))

        #import pdb;
        #pdb.set_trace()

        from scipy import stats

        # pval1 = 1 - stats.t.cdf(cars_t_test,df=len(cars))

        pval1 = 2 * (1 - stats.t.cdf(abs(cars_t_test), df=num_events))

        # pvalues = 2*(1-tcdf(abs(t),n-v))

        pval2 = 2 * (1 - stats.t.cdf(abs(cavs_t_test), df=num_events))

        if (pval1 < .05):
            cars_significant = True
        else:
            cars_significant = False

        if (pval2 < .05):
            cavs_significant = True
        else:
            cavs_significant = False

        #import pdb;
        #pdb.set_trace()

        ccr = CarsCavcsResult(num_events,
                              cars_cum, cars_std_err, cars_t_test, cars_significant,
                              cars_positive, cars_num_stocks_positive, cars_num_stocks_negative,
                              cavs_cum, cavs_std_err, cavs_t_test, cavs_significant,
                              cavs_positive, cavs_num_stocks_positive, cavs_num_stocks_negative)

        return ccr

        #import pdb;
        #pdb.set_trace()

    def calculate_using_single_factor_benchmark(self, event_matrix, stock_data, market_symbol, estimation_window=200,
                                                buffer=5,
                                                pre_event_window=10, post_event_window=10):
        '''

        :param event_matrix:
        :param stock_data:
        :param market_symbol:
        :param estimation_window:
        :param buffer:
        :param pre_event_window:
        :param post_event_window:
        :return cars_cavcs_result: An instance of CarsCavcsResult containing the results.


        Modeled after http://arno.uvt.nl/show.cgi?fid=129765

        '''

        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import datetime as dt

        # The event matrix has a row for every data in the stock data.
        # Zero (NaN) out any events in the rows at the beginning and end that would
        # not have data.
        starting_event_num = len(event_matrix[(event_matrix == 1.0).any(axis=1)])
        print("Starting number of events: {}".format(starting_event_num))
        event_matrix.values[0:estimation_window + buffer + pre_event_window + post_event_window, :] = np.NaN
        event_matrix.values[-estimation_window - buffer - pre_event_window - post_event_window:, :] = np.NaN

        ending_event_num = len(event_matrix[(event_matrix == 1.0).any(axis=1)])
        print("Ending number of events: {}".format(ending_event_num))
        if(starting_event_num != ending_event_num):
            print("{} events were dropped because they require data outside the data range.".format(starting_event_num -
                                                                                                    ending_event_num))

        events = event_matrix[(event_matrix == 1.0).any(axis=1)]



        dates = stock_data.loc[market_symbol, slice(None)].index
        date1 = events.index[0]
        index1 = dates.tolist().index(date1)
        wtf = dates[index1]
        date11 = dates[index1 - buffer]
        date12 = dates[index1 - (buffer + estimation_window)]

        #import pdb;
        #pdb.set_trace()

        # check remove duplicates

        stock_data.index.value_counts()
        stock_data.drop_duplicates(inplace=True)

        # import pdb; pdb.set_trace()

        try:

            # For IB
            closing_prices = stock_data['Close']
            volumes = stock_data['Volume']

        except KeyError:

            # For AV
            closing_prices = stock_data['adjusted_close']
            volumes = stock_data['volume']

        # check for duplicates
        closing_prices.index.value_counts()

        '''(RGR, 2005-12-30 00:00:00)    2
            (SPY, 2000-12-29 00:00:00)    2
            (RGR, 2006-12-29 00:00:00)    2'''

        # removing duplicates


        # now we are ready to do anlaysis

        stock_ret = closing_prices.copy()

        symbols = stock_data.index.get_level_values(0).unique().tolist()

        mypct = lambda x: x[-1] - np.mean(x[:-1])

        stock_ret = closing_prices.pct_change().fillna(0)

        vlm_changes = volumes.rolling(5, 5).apply(mypct).fillna(0)

        # do regeression

        pre_stock_returns = stock_ret[
            (stock_data.index.get_level_values(1) > date12) & (stock_data.index.get_level_values(1) <= date11)]

        pre_stock_vlms = vlm_changes[
            (stock_data.index.get_level_values(1) > date12) & (stock_data.index.get_level_values(1) <= date11)]

        # **************
        # First compute cars ******
        # ***************

        #import pdb;
        #pdb.set_trace()

        dates = stock_data.index.get_level_values(1).unique().tolist()

        if (market_symbol in symbols):
            stocks = [x for x in symbols if x != market_symbol]
        else:
            raise ValueError('calculate_using_single_factor_benchmark: market_symbol not found in data')

        ar1 = ['cars', 'cavs'];
        ar2 = ['slope', 'intercept']

        from itertools import product
        tuples = [(i, j) for i, j in product(ar1, ar2)]  # tuples = list(zip(*arrays))

        index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])

        df_regress = pd.DataFrame(0.0, index=index, columns=symbols)

        # import pdb; pdb.set_trace()

        for stock in stocks:
            # set up data

            x1 = pre_stock_returns[market_symbol]

            y1 = pre_stock_returns[stock]

            slope1, intercept1, cars0 = regress_vals(x1, y1)
            cars = np.cumprod(cars0 + 1, axis=0)

            # plot if you need

            #plot_regressvals(x1, y1, slope1, intercept1, cars, stock)

            # the same for cvals

            x2 = pre_stock_vlms[market_symbol]

            y2 = pre_stock_vlms[stock]

            # y2.argsort()[::-1][:n]


            # import pdb; pdb.set_trace()

            slope2, intercept2, cavs0 = regress_vals(x2, y2)
            cavs = np.cumsum(cavs0)

            #plot_regressvals(x2, y2, slope2, intercept2, cavs, stock)

            # store the regresion values


            df_regress.loc[('cars', 'slope'), stock] = slope1

            df_regress.loc[('cars', 'intercept'), stock] = intercept1

            df_regress.loc[('cavs', 'slope'), stock] = slope2

            df_regress.loc[('cavs', 'intercept'), stock] = intercept2


            # do the same for volumes

        # ***************
        # now the event cars and cavs computations

        ar11 = stocks
        ar12 = ['cars', 'cavs']

        tuples2 = [(i, j) for i, j in product(ar11, ar12)]  # tuples = list(zip(*arrays))

        index2 = pd.MultiIndex.from_tuples(tuples2, names=['first', 'second'])

        df_results = pd.DataFrame(0.0, index=index2, columns=['positive', 'significant'])

        ccarray = []
        cvarray = []

        # now the big loop

        #import pdb;
        #pdb.set_trace()

        try:

            for stock in stocks:

                slope1 = df_regress.loc[('cars', 'slope'), stock]
                intercept1 = df_regress.loc[('cars', 'intercept'), stock]

                slope2 = df_regress.loc[('cavs', 'slope'), stock]
                intercept2 = df_regress.loc[('cavs', 'intercept'), stock]

                ccr = []
                cvr = []

                for event in events.iterrows():
                    dt1 = event[0]
                    idx1 = dates.index(dt1)
                    window = dates[idx1 - pre_event_window:idx1 + post_event_window + 1]

                    dummy_rets = pd.Series(0.0, index=window)
                    event_rets = stock_ret.loc[stock, window].loc[stock, slice(None)]
                    event_mkt_ret = stock_ret.loc[market_symbol, window].loc[market_symbol, slice(None)]

                    # calculate excess returns

                    event_ex_ret = event_rets.subtract(slope1 * event_mkt_ret + intercept1, fill_value=0.0)

                    event_ex_ret = event_ex_ret.subtract(dummy_rets, fill_value=0.0)

                    event_cum_rets = np.cumprod(event_rets + 1, axis=0)
                    # plot_regressvals(event_mkt_ret,event_rets,slope1, intercept1,event_cum_rets,stock)

                    ccr.append((event_ex_ret.tolist()))

                    # cars = (event_cum_rets[-1] - 1) / len(window)

                    # now for vols
                    event_vols = vlm_changes.loc[stock, window].loc[stock, slice(None)]
                    mkt_vols = vlm_changes.loc[market_symbol, window].loc[market_symbol, slice(None)]

                    event_ex_vols = event_vols.subtract(slope2 * mkt_vols + intercept2, fill_value=0.0)
                    event_ex_vols = event_ex_vols.subtract(dummy_rets, fill_value=0.0)

                    event_cum_ex_vols = np.cumsum(event_ex_vols)
                    # plot_regressvals(mkt_vols,event_ex_vols,slope2, intercept2,event_cum_ex_vols,stock)

                    cvr.append(event_ex_vols.tolist())

                # *********************
                # now do computations for the whole stock

                #import pdb;
                #pdb.set_trace()

                # print(ccr)
                cars_stock = np.array([np.array(c) for c in ccr])

                ccarray.append(cars_stock)
                # df_results.loc[(slice(None),stock),'cars'].tolist()
                cars = np.mean(cars_stock, axis=0)

                std1 = np.std(cars)

                cars_t_test = np.mean(cars) / std1 * np.sqrt(len(window))

                # pval1 = 1 - stats.t.cdf(cars_t_test, df=len(cars))

                pval1 = 2 * (1 - stats.t.cdf(abs(cars_t_test), df=len(cars_stock)))

                if (pval1 < .05):
                    cars_significant = True
                else:
                    cars_significant = False

                if np.mean(cars) >= 0:
                    cars_positive = True
                else:
                    cars_positive = False

                cars_cum = np.cumprod(cars + 1, axis=0)

                #import pdb;
                #pdb.set_trace()

                # plt.plot(cars_cum); plt.title('Cars'); plt.show()

                # do the same for volumes
                # ***************

                cavs_stock = np.array(cvr)

                cvarray.append(cavs_stock)

                cavs = np.mean(cavs_stock, axis=0)

                std2 = np.std(cavs)

                cavs_t_test = np.mean(cavs) / std2 * np.sqrt(len(window))

                # pval2 = 1 - stats.t.cdf(cavs_t_test, df=len(cavs))

                pval2 = 2 * (1 - stats.t.cdf(abs(cavs_t_test), df=len(cavs_stock)))

                if (pval2 < .05):
                    cavs_significant = True
                else:
                    cavs_significant = False

                if np.mean(cavs) >= 0:
                    cavs_positive = True
                else:
                    cavs_positive = False

                cavs_cum = np.cumsum(cavs, axis=0)

                # import pdb; pdb.set_trace()


                # plt.plot(cavs_cum); plt.title('Cavs'); plt.show()


                #  store the results *******

                df_results.loc[(stock, 'cars'), 'positive'] = cars_positive

                df_results.loc[(stock, 'cars'), 'significant'] = cars_significant

                df_results.loc[(stock, 'cavs'), 'positive'] = cavs_positive

                df_results.loc[(stock, 'cavs'), 'significant'] = cavs_significant

        except Exception as e:
            #import pdb;
            #pdb.set_trace()
            #print(e)
            raise

        import pdb;
        #pdb.set_trace()

        # aggregate results for output
        # ****************

        positive1 = df_results.loc[(slice(None), 'cars'), 'positive'].tolist()

        significant1 = df_results.loc[(slice(None), 'cars'), 'significant'].tolist()

        cars_num_stocks_positive = sum(positive1)
        cars_num_stocks_negative = sum(np.logical_not(positive1))

        cars_num_stocks_significant = sum(significant1)

        positive2 = df_results.loc[(slice(None), 'cavs'), 'positive'].tolist()

        significant2 = df_results.loc[(slice(None), 'cavs'), 'significant'].tolist()

        cavcs_num_stocks_positive = sum(positive2)
        cavcs_num_stocks_negative = sum(np.logical_not(positive2))

        cavcs_num_stocks_significant = sum(significant2)

        # The full calculations *********

        # import pdb; pdb.set_trace()

        Cars = np.mean(np.array(ccarray), axis=0)

        num_events = len(Cars)

        cars_std_err = np.std(Cars, axis=0)

        cars = np.mean(Cars, axis=0)

        cars_cum = np.cumprod(cars + 1, axis=0)

        cars_t_testf = np.mean(Cars) / np.std(cars) * np.sqrt(len(window))

        # pval1 = 1 - stats.t.cdf(cars_t_testf, df=len(Cars))
        pval1 = 2 * (1 - stats.t.cdf(abs(cars_t_testf), df=num_events))

        if (pval1 < .05):
            cars_significant = True
        else:
            cars_significant = False

        if np.mean(cars) > 0:
            cars_positive = True
        else:
            cars_positive = False

        # ***********
        # Now cavs ******
        # *************

        # import pdb; pdb.set_trace()
        Cavcs = np.mean(np.array(cvarray), axis=0)

        cavcs_std_err = np.std(Cavcs, axis=0)

        cavcs = np.mean(Cavcs, axis=0)

        cavcs_cum = np.cumsum(cavcs, axis=0)

        cavcs_t_testf = np.mean(Cavcs) / np.std(cavcs) * np.sqrt(len(window))

        # pval2 = 1 - stats.t.cdf(cavcs_t_testf, df=len(Cavcs))
        pval2 = 2 * (1 - stats.t.cdf(abs(cavcs_t_testf), df=num_events))

        if (pval2 < .05):
            cavcs_significant = True
        else:
            cavcs_significant = False

        if np.mean(cavcs) > 0:
            cavcs_positive = True
        else:
            cavcs_positive = False

        # Final  Results to CarsCavcsResult

        #import pdb;
        #pdb.set_trace()
        ccr = CarsCavcsResult(num_events,
                              cars_cum, cars_std_err, cars_t_testf, cars_significant,
                              cars_positive, cars_num_stocks_positive, cars_num_stocks_negative,
                              cavcs_cum, cavcs_std_err, cavcs_t_testf, cavcs_significant,
                              cavcs_positive, cavcs_num_stocks_positive, cavcs_num_stocks_negative)

        return ccr


def plot_regressvals(x, y, slope, intercept, cars, stock):
    import matplotlib.pyplot as plt
    # import pdb; pdb.set_trace()
    plt.figure(1)
    ax1 = plt.subplot(211)
    plt.title('Regression for stock: ' + stock)
    ax1.plot(x, y, 'o', label='Original data', markersize=10)
    ax1.plot(x, slope * x + intercept, 'r', label='Fitted line')
    ax1.legend()

    ax2 = plt.subplot(212)
    ax2.plot(cars, label='excess return')
    plt.show()


def regress_vals(x, y):
    import numpy as np

    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

    """
    try:
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    except Exception as e:
        import pdb;
        pdb.set_trace()
        print(e)
    """

    # print(slope, intercept)
    yhat = slope * x + intercept
    cars0 = y - yhat
    # cars = np.cumprod(cars0 + 1, axis=0)

    return slope, intercept, cars0


class Plotter(object):
    def __init__(self, width=10, height=5):
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['grid.linestyle'] = "--"
        plt.rcParams['figure.figsize'] = width, height
        pass

    def plot_car(self, car, std_err, num_events, look_back, look_forward, show=True, pdf_filename=None):

        # plotting
        li_time = list(range(-look_back, look_forward + 1))
        # print(li_time)

        # Plotting the chart
        plt.clf()
        plt.grid()
        plt.axhline(y=1.0, xmin=-look_back, xmax=look_forward, color='k')
        plt.errorbar(li_time[look_back:], car[look_back:],
                     yerr=std_err[look_back:], ecolor='#AAAAFF',
                     alpha=0.7)
        plt.plot(li_time, car, linewidth=1, label='mean', color='b')
        plt.xlim(-look_back - 1, look_forward + 1)
        plt.title('Market Relative CAR of ' + str(num_events) + ' events')
        plt.xlabel('Days')
        plt.ylabel('Cumulative Abnormal Returns')
        if pdf_filename is not None:
            plt.savefig(pdf_filename, format='pdf')
        if show:
            plt.show()

    def plot_car_cavcs(self, num_events, car, std_err1, cavcs, std_err2, look_back, look_forward, show=True,
                       pdf_filename=None):

        # printing some Output
        li_time = list(range(-look_back, look_forward + 1))
        # print(li_time)



        # Plotting the chart first for cavcs
        plt.clf()
        plt.figure(1)
        ax1 = plt.subplot(211)
        plt.grid()
        plt.axhline(y=1.0, xmin=-look_back, xmax=look_forward, color='k')
        plt.errorbar(li_time[look_back:], car[look_back:],
                     yerr=std_err1[look_back:], ecolor='#AAAAFF',
                     alpha=0.7)
        ax1.plot(li_time, car, linewidth=1, label='mean', color='b')
        plt.xlim(-look_back - 1, look_forward + 1)
        plt.title('Market Relative CAR & CAVCS of ' + str(num_events) + ' events')
        plt.xlabel('Days')
        plt.ylabel('Cumulative Abnormal Returns')

        # now the cavcs

        ax2 = plt.subplot(212)
        plt.grid()
        plt.axhline(y=1.0, xmin=-look_back, xmax=look_forward, color='k')
        plt.errorbar(li_time[look_back:], cavcs[look_back:],
                     yerr=std_err2[look_back:], ecolor='#AAAAFF',
                     alpha=0.7)
        ax2.plot(li_time, cavcs, linewidth=1, label='mean', color='b')
        plt.xlim(-look_back - 1, look_forward + 1)
        plt.xlabel('Days')
        plt.ylabel('Cumulative Abnormal Volumes Changes')

        # write it our
        if pdf_filename is not None:
            plt.savefig(pdf_filename, format='pdf')
        if show:
            plt.show()




