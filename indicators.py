

from collections import deque
import pandas
import numpy
import statistics
from datetime import datetime, timedelta
import math
import calendar_calcs


def to_date(date_object):
    if isinstance(date_object, str):
        return datetime.strptime(date_object, "%Y-%m-%d").date()
    return date_object

def _weekday(date_object):
    return to_date(date_object).weekday()

## consumes an entire dataframe and returns
## a parallel timeseries of the desired indicator
def indicator_to_df(stock_df, indicator, name='Value', merge=False):

    derived = []
    for i in range(stock_df.shape[0]):
        index = stock_df.index[i]
        stock_bar = stock_df.loc[index]
        v = indicator.push(stock_bar)
        if v is None: v = numpy.nan
        derived.append( {'Date':stock_bar['Date'], name:v })

    new_df = pandas.DataFrame(derived)
    if merge:
        new_df = pandas.merge(stock_df, new_df, on='Date', how='left')

    return new_df


class Indicator(object):
    def __init__(self, history_len, derived_len=None):
        self.history = deque()
        self.derived = deque()
        self.history_len = history_len
        self.derived_len = derived_len
        self.pushed = 0

    def push(self, data, valueAt=0):
        t_data = self.transform(data)
        self.history.append(t_data)
        self.pushed += 1

        old_len = len(self.derived)

        ## flag when new derived data is available 
        self._calculate()
        new_len = len(self.derived)

        if self.derived_len and len(self.derived) > self.derived_len: self.derived.popleft()
        if len(self.history) > self.history_len: self.history.popleft()


        if new_len > old_len:
            ## return requested history value, default is the current value: self.valueAt(0)
            return self.valueAt(idx=valueAt)
        else:
            return None

    def count(self):
        return len(self.derived)

    ## external mechanism to attach at transform function
    ## without having to write a new indicator
    def attach_transform(self, tfunc):
        self.transform = tfunc

    def transform(self, data_point):
        #transform historical data point before feeding it to indicator calcs
        #this allows us to pick of a value in a dataframe, combine dataframe values, etc..
        #example - see EMA versus CloseEMA:
        #EMA works on a single datapoint value, where CloseEMA provides that specialized datapoint as df_bar['Close']

        return data_point

    def _calculate(self):
        return None 

    def valueAt(self, idx):
        if idx >= 0 and len(self.derived) >= idx+1:
            i = -1
            if idx > 0: i = -(idx+1)
            return self.derived[i]
        else:
            return None


def cross_up(timeseries, threshold, front_index, back_index):
    x_up = False
    front = back = None
    if isinstance(threshold, Indicator):
        front = threshold.valueAt(front_index)
        back = threshold.valueAt(back_index)
    else:
        ## constant value
        front = back = threshold
        
    if front != None and back != None:
        if timeseries.valueAt(front_index) > front:
            if timeseries.valueAt(back_index) <= back: 
                x_up = True

    return x_up

def cross_dwn(timeseries, threshold, front_index, back_index):
    x_dwn = False
    front = back = None
    if isinstance(threshold, Indicator):
        front = threshold.valueAt(front_index)
        back = threshold.valueAt(back_index)
    else:
        ## constant value
        front = back = threshold
        
    if front != None and back != None:
        if timeseries.valueAt(front_index) < front:
            if timeseries.valueAt(back_index) >= back: 
                x_dwn = True

    return x_dwn


class DataSeries(Indicator):
    def __init__(self, derived_len=50):
        super().__init__(history_len=0, derived_len=derived_len)

    def _calculate(self):
        self.derived.append(self.history[-1])


class WeeklyBar(Indicator):
    def __init__(self):
        super().__init__(history_len=10)
        self.holidays = calendar_calcs.load_holidays() 
        self.FRIDAY = 4
        self.open = self.high = self.low = self.close = None
        self.volume = 0

    def _clear_week(self):
        self.open = self.high = self.low = self.close = None
        self.volume = 0 

    def get_week(self, data_dt):
        ONE_WEEK = 7
        week  = 0
        prev = data_dt
        while prev.month == data_dt.month:
            prev -= timedelta(days=ONE_WEEK)
            week += 1
        return week

    def _calculate(self):
        ## expecting a OHLC bar to be pushed
        daily_bar = self.history[-1]

        if self.open is None: self.open = daily_bar['Open']
        if self.high is not None:
            self.high = max(self.high, daily_bar['High'])
        else:
            self.high = daily_bar['High']
        if self.low is not None:
            self.low = min(self.low, daily_bar['Low'])
        else:
            self.low = daily_bar['Low']
        self.close = daily_bar['Close']
        self.volume += daily_bar['Volume']

        data_dt = datetime.strptime(daily_bar['Date'],"%Y-%m-%d").date()
        if calendar_calcs.is_end_of_week(data_dt, self.holidays):
            v = { 
                    'Date':daily_bar['Date'],
                    'Week':self.get_week(data_dt),
                    'Open':self.open,
                    'High':self.high,
                    'Low':self.low,
                    'Close':self.close,
                    'Volume':self.volume
            }
            self.derived.append(v)
            self._clear_week()

    def convert_daily_dataframe(self, stock_df):
        for i in range(stock_df.shape[0]):
            idate = stock_df.index[i]
            stock_bar = stock_df.loc[idate]
            self.push(stock_bar)

        return pandas.DataFrame(self.derived)


class Mo(Indicator):
    def __init__(self, history_len, derived_len=50):
        super().__init__(history_len, derived_len)

    def _calculate(self):
        if len(self.history) >= self.history_len:
            n = self.history[-1]
            p = self.history[-self.history_len]
            m = n - p
            self.derived.append(m)

class RunMonitor(Indicator):
    def __init__(self, derived_len=50):
        super().__init__(history_len=2, derived_len=derived_len)
        self.downs = self.ups = 0
        self.up_total = self.down_total = 0

    def _calculate(self):
        ## creates tuples representing consecutive changes (runs) in a time series -> (run_count, magnitude)
        ## negative values indicate consecutive down moves: (-3, -4,5) = down 3 days with total maginitude of -4.5 points
        ## positive values indicate consecutive up moves: (5, 11.24) = up 5 days with total maginitude of 11.24 points
        ## zero price changes are neutral
        if len(self.history) >= self.history_len:
            n = self.history[-1]
            p = self.history[-2]
            m = n - p
            if m < 0:
                self.downs -= 1
                self.down_total += m
                self.ups = self.up_total  = 0 
                self.derived.append((self.downs, self.down_total))
            elif m > 0:
                self.ups += 1
                self.up_total += m
                self.downs = self.down_total  = 0
                self.derived.append((self.ups, self.up_total))
            else:
                if len(self.derived) > 0:
                    prev_cnt, prev_val = self.derived[-1]
                    if prev_cnt < 0: prev_cnt -= 1
                    if prev_cnt > 0: prev_cnt += 1
                    self.derived.append((prev_cnt, prev_val))
                else:
                    self.derived.append((0,0))

class LogRtn(Indicator):
    def __init__(self, history_len, derived_len=50):
        super().__init__(history_len, derived_len)

    def _calculate(self):
        if len(self.history) >= self.history_len:
            n = self.history[-1]
            p = self.history[-self.history_len]
            m = math.log(n/p)
            self.derived.append(m)

class MA(Indicator):
    def __init__(self, history_len, derived_len=50):
        super().__init__(history_len, derived_len=derived_len)

    def _calculate(self):
        if len(self.history) >= self.history_len:
            m = list(self.history)[-self.history_len:]
            self.derived.append(statistics.mean(m))


class IBS(Indicator):
    ## internal bar strength iindicator
    ## (Close - Low)/(High - Low)
    def __init__(self, derived_len=50):
        super().__init__(history_len=1, derived_len=derived_len)

    def _calculate(self):
        if len(self.history) >= self.history_len:
            bar = self.history[-1]
            ibs = (bar['Close'] - bar['Low'])/(bar['High'] - bar['Low']) * 100
            self.derived.append(ibs)

class RSI(Indicator):
    def __init__(self, history_len, derived_len=50):
        super().__init__(history_len, derived_len)
        self.INF = 1e10
        self.upv = None
        self.dnv = None

    def _calculate(self):
        if len(self.history) >= self.history_len+1:
            # subtract current prices a[1:], from last prices a[:-1]
            # remember the most current price is the last price in the deque/list
            chgs = [x[0] - x[1] for x in zip(list(self.history)[1:],list(self.history)[:-1])]
            ups = sum([ x for x in chgs if x >= 0 ])
            dns = sum([ abs(x) for x in chgs if x < 0 ])

            a = 1.0/self.history_len
            upv = a * ups + (1-a) * self.upv if self.upv is not None else ups 
            dnv = a * dns + (1-a) * self.dnv if self.dnv is not None else dns
            self.upv, self.dnv = upv, dnv
            if dnv == 0: dnv = self.INF

            rsi = 100.0 - 100.0/(1+(upv/dnv))
            self.derived.append(rsi)

class CutlersRSI(Indicator):
    def __init__(self, history_len, derived_len=50):
        super().__init__(history_len, derived_len)

    def _calculate(self):
        if len(self.history) >= self.history_len+1:
            # subtract current prices a[1:], from last prices a[:-1]
            # remember the most current price is the last price in the deque/list
            chgs = [x[0] - x[1] for x in zip(list(self.history)[1:],list(self.history)[:-1])]
            ups = sum([ x for x in chgs if x >= 0 ])
            dns = sum([ abs(x) for x in chgs if x < 0 ])

            rsi = None
            if dns == 0:
                rsi = 100 
            else:
                rsi = 100.0 - 100.0/(1+(ups/dns))

            self.derived.append(rsi)

            
class Thanos(Indicator):
    def __init__(self, ma_len, no_of_samples, derived_len=50):
        # ma_len = benchmark moving average
        # no_of_samples = cnt of devations vs the ma needed to calculate zscore 
        super().__init__(history_len=(ma_len+no_of_samples), derived_len=derived_len)
        self.deviations = deque()
        self.ma_len = ma_len
        self.no_of_samples = no_of_samples

    def _calculate(self):
        if len(self.history) >= self.ma_len:
            ma = statistics.mean(list(self.history)[-self.ma_len:])
            dev = math.log(self.history[-1]/ma)
            self.deviations.append(dev)
            self.history.popleft()

            if len(self.deviations) >= self.no_of_samples:
                samples = list(self.deviations)[-self.no_of_samples:]
                z = (dev - statistics.mean(samples))/statistics.pstdev(samples)
                self.derived.append(z)
                self.deviations.popleft()


class StDev(Indicator):
    def __init__(self, sample_size, derived_len=50):
        super().__init__(history_len=sample_size, derived_len=derived_len)

    def _calculate(self):
        if len(self.history) >= self.history_len:
            m = list(self.history)[-self.history_len:]
            v = pandas.Series(data=m)
            self.derived.append(v.std())

class Corr(Indicator):
    ## correlation - expects pairs to be pushed (price1, price2)
    ## calculates the correlation of the returns of time series price1 vs time series price2
    def __init__(self, sample_size, derived_len=50):
        super().__init__(history_len=sample_size+1, derived_len=derived_len)

    def returns(self, price_array):
        p = price_array
        rtns = []
        for i in range(1, len(p)):
            r = math.log(p[i]/p[i-1])
            rtns.append(r)
        return rtns

    def _calculate(self):
        if len(self.history) >= self.history_len:
            pairs = list(self.history)[-self.history_len:]
            a = pandas.Series(data= self.returns([ x[0] for x in pairs ]))
            b = pandas.Series(data= self.returns([ x[1] for x in pairs ]))
            m = a.corr(b)
            self.derived.append(m)

class Beta(Indicator):
    ## beta - expects pairs to be pushed (price1, price2)
    ## calculates the beta of the returns of time series price1 vs time series price2
    def __init__(self, sample_size, derived_len=50):
        super().__init__(history_len=sample_size+1, derived_len=derived_len)

    def returns(self, price_array):
        p = price_array
        rtns = []
        for i in range(1, len(p)):
            r = math.log(p[i]/p[i-1])
            rtns.append(r)
        return rtns

    def _calculate(self):
        if len(self.history) >= self.history_len:
            pairs = list(self.history)[-self.history_len:]
            a = pandas.Series(data= self.returns([ x[0] for x in pairs ]))
            b = pandas.Series(data= self.returns([ x[1] for x in pairs ]))
            m = a.cov(b)/b.var()
            self.derived.append(m)

class Median(Indicator):
    def __init__(self, sample_size, derived_len=50):
        super().__init__(history_len=sample_size, derived_len=derived_len)
        self.sample_sz = sample_size

    def _calculate(self):
        if len(self.history) >= self.history_len:
            m = statistics.median(list(self.history)[-self.sample_sz:])
            self.derived.append(m)

class ZScore(Indicator):
    def __init__(self, sample_size, derived_len=50):
        super().__init__(history_len=sample_size+1, derived_len=derived_len)
        self.sample_sz = sample_size+1

    def _calculate(self):
        if len(self.history) >= self.history_len:
            pop = list(self.history)[-(self.sample_sz):-2]
            v = self.history[-1]
            s = statistics.pstdev(pop)
            m = statistics.mean(pop)
            self.derived.append((v-m)/s)


class EMA(Indicator):
    def __init__(self, coeff, warmup, history_len, derived_len=50):
        super().__init__(history_len, max(derived_len, warmup))
        self.coeff = coeff
        self.warmup = warmup
        self.counter = 0
        self.prev = None

    def _calculate(self):
        n = self.history[-1]
        if self.prev is not None:
            self.prev = (self.coeff * n) + (1 - self.coeff) * self.prev
        else:
            self.prev = n
        self.counter += 1

        if self.counter >= self.warmup:
            self.derived.append(self.prev)
            self.counter = self.warmup  #just to prevent potential rollover
            return self.prev
        else:
            return None

class LastLow(Indicator):
    def __init__(self, last_len, derived_len=50):
        super().__init__(last_len, derived_len)
        self.last_len = last_len

    def _calculate(self):
        if len(self.history) >= self.history_len:
            lowest = min(list(self.history)[-self.last_len:])
            self.derived.append(lowest)
            return lowest
        else:
            return None

class LastHigh(Indicator):
    def __init__(self, last_len, derived_len=50):
        super().__init__(last_len, derived_len)
        self.last_len = last_len

    def _calculate(self):
        if len(self.history) >= self.history_len:
            highest = max(list(self.history)[-self.last_len:])
            self.derived.append(highest)
            return highest 
        else:
            return None


class MACD(Indicator):
    def __init__(self, long_len, short_len, signal_len, warmup=10, history_len=50, derived_len=50):
        super().__init__(history_len, derived_len)
        self.q, self.k, self.sig  = 2.0/long_len, 2.0/short_len, 2.0/signal_len
        self.warmup = warmup 
        self.signal_warmup = signal_len
        self.counter = 0
        self.signal_counter = 0
        self.prev_q, self.prev_k, self.prev_sig = None, None, None

    def _calculate(self):
        n = self.history[-1]
        if self.prev_q is not None:
            self.prev_q = (self.q * n) + (1 - self.q) * self.prev_q
        else:
            self.prev_q = n
        if self.prev_k is not None:
            self.prev_k = (self.k * n) + (1 - self.k) * self.prev_k
        else:
            self.prev_k = n

        self.counter += 1

        if self.counter >= self.warmup:
            diff = self.prev_k - self.prev_q
            if self.prev_sig is not None:
                self.prev_sig = (self.sig * diff) + (1 - self.sig) * self.prev_sig
            else:
                self.prev_sig = diff
            
            self.counter = self.warmup  #just to prevent counter rollover
            self.signal_counter += 1

            macd = None 
            if self.signal_counter >= self.signal_warmup:
                macd = diff - self.prev_sig
                self.derived.append(macd)
                self.signal_counter = self.signal_warmup  #just to prevent counter rollover
            return macd
        else:
            return None


class CloseEMA(EMA):
    def __init__(self, coeff, warmup, history_len, derived_len=50):
        super().__init__(coeff, warmup, history_len, derived_len)

    def transform(self, df_bar):
        return df_bar['Close']


class CloseMACD(MACD):
    def __init__(self, long_len, short_len, signal_len, warmup=10, history_len=50, derived_len=50):
        super().__init__(long_len, short_len, signal_len, warmup, history_len, derived_len)

    def transform(self, df_bar):
        return df_bar['Close']

class WeightedClose(Indicator):
    def __init__(self, history_len, derived_len=50):
        super().__init__(history_len, derived_len)

    def _calculate(self):
        n = self.history[-1]
        wp = ( n['Close'] + n['High'] + n['Low'] )/3.0
        self.derived.append(wp)
        return self.derived[-1]

## uses first day of the week range as breakout points
## requires true OHLC bar information from a DataFrame and date as a tuple -> (dt, OHLC bar)
## pushes to self.derived (current anchor, breakout amt above/below anchor range)
class MondayAnchor(Indicator):
    def __init__(self, derived_len=50):
        super().__init__(history_len=1, derived_len=derived_len)
        self.holidays = calendar_calcs.load_holidays() 
        self.anchor = None

    def _calculate(self):
        ## expecting (datetime.date, OHLC bar) tuple
        dt, bar = self.history[-1]
        if calendar_calcs.is_start_of_week(dt, self.holidays):
            self.anchor = bar
            ## record new anchor with no breakout
            self.derived.append((self.anchor, 0))
        else:
            ## record breakouts above or below current anchor
            if self.anchor is not None:
                if bar['Close'] > self.anchor['High']:
                    self.derived.append((self.anchor, bar['Close'] - self.anchor['High']))
                elif bar['Close'] < self.anchor['Low']:
                    self.derived.append((self.anchor, bar['Close'] - self.anchor['Low']))
                else:
                    self.derived.append((self.anchor, 0))

class Anchor(Indicator):
    def __init__(self, day, derived_len=50):
        super().__init__(history_len=1, derived_len=derived_len)
        self.holidays = calendar_calcs.load_holidays()
        self.anchor = None
        self.day = day

    def _calculate(self):
        ## expecting (datetime.date, OHLC bar) tuple
        dt, bar = self.history[-1]
        if calendar_calcs.is_day_of_week(self.day, dt, self.holidays):
            self.anchor = bar
            ## record new anchor with no breakout
            self.derived.append((self.anchor, 0))
        else:
            ## record breakouts above or below current anchor
            if self.anchor is not None:
                if bar['Close'] > self.anchor['High']:
                    self.derived.append((self.anchor, bar['Close'] - self.anchor['High']))
                elif bar['Close'] < self.anchor['Low']:
                    self.derived.append((self.anchor, bar['Close'] - self.anchor['Low']))
                else:
                    self.derived.append((self.anchor, 0))


## uses last day of the week range as breakout points
class LastDayAnchor(Indicator):
    def __init__(self, derived_len=50):
        super().__init__(history_len=1, derived_len=derived_len)
        self.holidays = calendar_calcs.load_holidays() 
        self.anchor = None

    def _calculate(self):
        ## expecting (datetime.date, OHLC bar) tuple
        dt, bar = self.history[-1]
        if calendar_calcs.is_end_of_week(dt, self.holidays):
            self.anchor = bar
            ## record new anchor with no breakout
            self.derived.append((self.anchor, 0))
        else:
            ## record breakouts above or below current anchor
            if self.anchor is not None:
                if bar['Close'] > self.anchor['High']:
                    self.derived.append((self.anchor, bar['Close'] - self.anchor['High']))
                elif bar['Close'] < self.anchor['Low']:
                    self.derived.append((self.anchor, bar['Close'] - self.anchor['Low']))
                else:
                    self.derived.append((self.anchor, 0))


## returns a tuple of cardinal alue of current trading day (week, month, year)
## expecting successive dt from a progressing timeseries 
class TradingDayMarker(Indicator):
    def __init__(self, derived_len=None):
        super().__init__(history_len=1, derived_len=derived_len)
        self.prev_year = None
        self.prev_month = None
        self.wk = self.mth = self.yr = None

    def _calculate(self):
        ## expecting successive dt 
        dt = self.history[-1]
        mark_date = pandas.to_datetime(dt)
        self.wk = mark_date.weekday()
        month = mark_date.month
        year = mark_date.year

        if self.prev_month is None: self.prev_month = month
        if self.prev_year is None: self.prev_year = year 

        if month != self.prev_month:
            self.mth = 1
            self.prev_month = month
        elif self.mth is not None:
            self.mth += 1

        if year != self.prev_year:
            self.yr = 1
            self.prev_year = year
        elif self.yr is not None:
            self.yr += 1

        self.derived.append((self.wk, self.mth, self.yr))


class VolatilityStop(Indicator):
    def __init__(self, stdev_multiplier, default_stop, history_len, derived_len=50):
        super().__init__(history_len, derived_len)
        ## stdev_multiplier = what we mulitiply the volatility over history len that 
        ##     we use calculate the stop: current_high - volatility * stdev_mulitplier
        ## default_stop = default % loss allowed when stdev-derived stop value isn't available 
        self.stdev_multiplier = abs(stdev_multiplier)
        self.default_stop = abs(default_stop)
        self.highest = None
        self.first_open = None 
        self.current_stop = None
        self._stdev = deque() 
        self.position = 0


    def push(self, current_bar, position):
        ## propogate current position to calculate step
        self.position = position
        return super().push(current_bar)

    def _calc_stdev(self):
        if len(self.history) >= self.history_len:
            highs = [ bar['High'] for bar in self.history ]
            lows = [ bar['Low'] for bar in self.history ]
            ## do dispersion based on price range extremes 
            self._stdev.append(statistics.pstdev((highs + lows)))
            if len(self._stdev) > self.derived_len:
                self._stdev.popleft()
    
    def stdev(self, index=0):
        i = abs(index)+1
        if i <= len(self._stdev):
            return self._stdev[-i]
        else:
            return None
        
    def _calculate(self):
        current_bar = self.history[-1]

        self._calc_stdev()

        m = None
        if self.position > 0:
            prev_stdev = self.stdev(1)
            if self.first_open is None:
                self.first_open = current_bar['Open']
                if prev_stdev is None:
                    m = (1-self.default_stop) * self.first_open
                    #print(current_bar.name, f'Anchor (Open): {self.first_open} Stop= {m} Using default {self.default_stop}')
                else:
                    m = self.first_open - (prev_stdev * self.stdev_multiplier)
                    #print(current_bar.name, f'Anchor (Open): {self.first_open} Stop= {m} Using stdev {prev_stdev} x {self.stdev_multiplier}')
                ## grab trailing high value
                self.highest = current_bar['High']
            else:
                if prev_stdev is None:
                    m = (1-self.default_stop) * self.highest
                    #print(current_bar.name, f'Anchor (High): {self.highest} Stop= {m} Using default {self.default_stop}')
                else:
                    m = self.highest - (prev_stdev *  self.stdev_multiplier)
                    #print(current_bar.name, f'Anchor (High): {self.highest} Stop= {m} Using stdev {prev_stdev} x {self.stdev_multiplier}')
                self.highest = max(self.highest, current_bar['High'])
    
            if self.current_stop is None:
                self.current_stop = m
            else:
                self.current_stop = max(self.current_stop, m)
        else:
            self.highest = None
            self.first_open = None
            self.current_stop = None

        self.derived.append(self.current_stop)
        return self.current_stop 


class VolatilityStopClose(VolatilityStop):
    def __init__(self, stdev_multiplier, default_stop, history_len, derived_len=50):
        super().__init__(stdev_multiplier, default_stop, history_len, derived_len)

    def _calc_stdev(self):
        if len(self.history) >= self.history_len:
            close_prices = [ bar['Close'] for bar in self.history ]
            ## do dispersion based on price range extremes 
            self._stdev.append(statistics.pstdev(close_prices))
            if len(self._stdev) > self.derived_len:
                self._stdev.popleft()


def create_simulated_timeseries(length):
    from datetime import date, timedelta
    from random import randint

    df = None
    opn = 100
    dt = date(2022,5,10)
    for i in range(length):
        hi = opn + randint(1,100)/100.0
        lo = opn - randint(1,100)/100.0
        r = (hi-lo)
        close = (r * randint(1,10)/100.0) + lo
        bar = pandas.DataFrame(columns='Date Open High Low Close Day'.split(),data=[[dt,opn,hi,lo,close,dt.weekday()]])
        opn = hi - (r * randint(1,10)/100.0)
        dt = dt+timedelta(days=3) if dt.weekday() == 4 else dt+timedelta(days=1)
        if df is None:
            df = pandas.DataFrame(bar)
        else:
            df = pandas.concat([df,bar])

    df.set_index('Date', inplace=True)

    return df


def test_monday_anchor():

    df = create_simulated_timeseries(length=16)
    print(df)

    m = MondayAnchor()
    for i in range(df.shape[0]):
        cur_dt = df.index[i]
        v = m.push((cur_dt, df.loc[cur_dt]))
        print("bar = ", df.loc[cur_dt])
        print(v)
        print("")


def test_ema():

    e = EMA(coeff=0.3, warmup=10, history_len=50, derived_len=20)

    from random import randint
    changes = [ randint(-100,100)/100.0 for x in range(100) ]
    changes[0] = 100
    prices = [ sum(changes[0:i]) for i in range(1,101) ]
    for i, p in enumerate(prices):
        v = e.push(p)
        print(f'{i:03d} {p:10.4f} {v}')

    print(e.valueAt(2))
    print(e.valueAt(1))
    print(e.valueAt(0))

def test_correlation():

    e = Corr(sample_size=6)
   
    prices = [ (43, 99), (21, 65), (25,79), (42,75), (57,87), (59,81) ]
    for i, p in enumerate(prices):
        v = e.push(p)
        print(f'{i:03d} {p} {v}')

def test_beta():

    e = Beta(sample_size=6)
   
    prices = [ (43, 99), (21, 65), (25,79), (42,75), (57,87), (59,81) ]
    for i, p in enumerate(prices):
        v = e.push(p)
        print(f'{i:03d} {p} {v}')

def test_stdev():

    e = StDev(sample_size=6)
   
    prices = [ (43, 99), (21, 65), (25,79), (42,75), (57,87), (59,81) ]
    for i, p in enumerate(prices):
        v = e.push(p[0])
        print(f'{i:03d} {p} {v}')

    m = [x[0] for x in prices]
    j = pandas.Series(data=m)
    print(j.std())

def test_zscore():

    e = ZScore(sample_size=50)

    from random import randint
    changes = [ randint(-100,100)/100.0 for x in range(60) ]
    changes[0] = 100
    prices = [ sum(changes[0:i]) for i in range(1,61) ]
    for i, p in enumerate(prices):
        v = e.push(p)
        print(f'{i:03d} {p:10.4f} {v}')


def test_macd():

    e = MACD(long_len=28, short_len=12, signal_len=9, history_len=50, derived_len=20)

    from random import randint
    changes = [ randint(-100,100)/100.0 for x in range(100) ]
    changes[0] = 100
    prices = [ sum(changes[0:i]) for i in range(1,101) ]
    for i, p in enumerate(prices):
        v = e.push(p)
        #print(f'{i:03d} {p:10.4f} {v}')

def test_cross_up():

    e = MA(9)
    d = DataSeries()

    from random import randint
    changes = [ randint(-100,100)/100.0 for x in range(100) ]
    changes[0] = 100
    prices = [ sum(changes[0:i]) for i in range(1,101) ]
    for i, p in enumerate(prices):
        v = e.push(p)
        x = d.push(p)
        x_up = cross_up(d,e,0,1)
        above = ''
        if v is not None and x is not None and v < x: above = 'ABOVE'
        print(f'{i:03d} {p:10.4f} ma={v}   {x_up}  {above}')

def test_cross_dwn():

    e = MA(9)
    d = DataSeries()

    from random import randint
    changes = [ randint(-100,100)/100.0 for x in range(100) ]
    changes[0] = 100
    prices = [ sum(changes[0:i]) for i in range(1,101) ]
    for i, p in enumerate(prices):
        v = e.push(p)
        x = d.push(p)
        x_dn = cross_dwn(d,e,0,1)
        below = ''
        if v is not None and x is not None and v > x: below = 'BELOW'
        print(f'{i:03d} {p:10.4f} ma={v}   {x_dn}  {below}')


def test_last_low():

    e = LastLow(last_len=5)

    from random import randint
    changes = [ randint(-100,100)/100.0 for x in range(20) ]
    changes[0] = 100
    prices = [ sum(changes[0:i]) for i in range(1,21) ]
    for i, p in enumerate(prices):
        v = e.push(p)
        print(f'{i:03d} {p:10.4f} {v}')
    print(len(e.history))

def test_rsi():

    e = RSI(14)

    from random import randint
    samples = 500
    changes = [ randint(-300,300)/100.0 for x in range(samples) ]
    changes[0] = 100
    prices = [ sum(changes[0:i]) for i in range(1,samples+1) ]
    for i, p in enumerate(prices):
        v = e.push(p)
        print(f'{i:03d} {p:10.4f} {v}')
    print(len(e.history))

def test_cutlers_rsi():

    e = CutlersRSI(3)

    from random import randint
    samples = 20
    changes = [ randint(-300,300)/100.0 for x in range(samples) ]
    changes[0] = 100
    prices = [ sum(changes[0:i]) for i in range(1,samples+1) ]
    for i, p in enumerate(prices):
        v = e.push(p)
        print(f'{i:03d} {p:10.4f} {v}')
    print(len(e.history))


def test_dataseries():

    e = DataSeries(14)

    from random import randint
    samples = 50
    changes = [ randint(-300,300)/100.0 for x in range(samples) ]
    changes[0] = 100
    prices = [ sum(changes[0:i]) for i in range(1,samples+1) ]
    for i, p in enumerate(prices):
        v = e.push(p)
        print(f'{i:03d} {p:10.4f} {v}')
    print(e.derived)
    print(e.valueAt(3))

def test_runs():

    e = RunMonitor()

    from random import randint
    samples = 50
    changes = [ randint(-300,300)/100.0 for x in range(samples) ]
    changes[0] = 100
    prices = [ sum(changes[0:i]) for i in range(1,samples+1) ]
    prev = None
    for i, p in enumerate(prices):
        v = e.push(p)
        if prev is not None:
            c = p - prev
            print(f'{i:03d} {p:10.4f} {c:10.4f} {v}')
        prev = p 


def test_thanos():

    e = Thanos(ma_len=50, no_of_samples=20)

    from random import randint
    samples = 500
    changes = [ randint(-300,300)/100.0 for x in range(samples) ]
    changes[0] = 100
    prices = [ sum(changes[0:i]) for i in range(1,samples+1) ]
    for i, p in enumerate(prices):
        v = e.push(p)
        print(f'{i:03d} {p:10.4f} {v}')


def test_volatility_stop():
    from random import randint
    df = create_simulated_timeseries(length=200)

    ## create a simulated position series 
    ## and tack it on the side of the price timeseries
    q = []
    start = 10
    stop = start+ randint(60,100)
    ## create a random on/off long position
    for i in range(len(df)):
        p = 0
        if i >= start and i <stop:
            p = 1
        if i == stop:
            start = stop + randint(10,60)
            stop = start + randint(10,60)
        q.append(p)

    pos = pandas.Series(q,index=df.index)
    df['Pos'] = pos

    v = VolatilityStop(stdev_multiplier=2.0, default_stop=0.3, history_len=10, derived_len=50)

    stops = []
    stdevs = []
    for i in range(len(df)):
        bar = df.loc[df.index[i]]
        stops.append(v.push(bar,bar['Pos']))
        stdevs.append(v.stdev())

    ss = pandas.Series(stops,index=df.index)
    yy = pandas.Series(stdevs,index=df.index)

    df['StDev'] = yy
    df['Stop'] = ss

    df.to_html('voltest_table.html')
        



if __name__ == '__main__':
    #test_thanos()
    #test_rsi()
    #test_cutlers_rsi()
    #test_dataseries()
    #test_monday_anchor()
    #test_cross_dwn()
    #test_ema()
    test_correlation()
    test_stdev()
    test_beta()
    #test_macd()
    #test_converter()
    #test_converter_with_indicator()
    #test_converter_with_ema()
    #test_volatility_stop()
    #test_last_low()
    #test_runs()



