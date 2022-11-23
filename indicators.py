

from collections import deque
import pandas
import statistics
from datetime import datetime
import math
import calendar_calcs


def to_date(date_object):
    if isinstance(date_object, str):
        return datetime.strptime(date_object, "%Y-%m-%d").date()
    return date_object

def _weekday(date_object):
    return to_date(date_object).weekday()

class Indicator(object):
    def __init__(self, history_len, derived_len):
        self.history = deque()
        self.derived = deque()
        self.history_len = history_len
        self.derived_len = derived_len
        self.pushed = 0

    def push(self, data, valueAt=0):
        t_data = self.transform(data)
        self.history.append(t_data)
        self.pushed += 1
        self._calculate()

        if len(self.derived) > self.derived_len: self.derived.popleft()
        if len(self.history) > self.history_len: self.history.popleft()
        if len(self.derived) > 0:
            ## return requested history value, default is the current value: self.valueAt(0)
            return self.valueAt(idx=valueAt)
        else:
            return None

    def count(self):
        return len(self.derived)

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


class WeeklyConverter(object):
    def __init__(self, daily_df, indicator_dict = None):
        self.FRIDAY = 4
        self.daily_df = daily_df
        self.weekly_df = None
        self.open, self.high, self.low, self.close = None, None, None, None

        self.indicator_dict = dict()
        if indicator_dict is not None:
            self.indicator_dict = indicator_dict

        self.last_week_bar = self._empty_bar()

    def _empty_bar(self):
        ## include indicator columns if given
        inames = []
        if self.indicator_dict is not None:
            inames = sorted(self.indicator_dict.keys())

        cols = 'Date Open High Low Close'.split() + inames + ['EndOfWeek']
        empties = [None for i in range(len(cols))]
        return pandas.DataFrame(columns=cols, data=[empties])


    def _fill_bar(self, dt, is_end_of_week):
        df = self._empty_bar()

        df['Date'] = dt
        df['Open'] = self.open
        df['High'] = self.high
        df['Low'] = self.low
        df['Close'] = self.close
        df['EndOfWeek'] = is_end_of_week

        # fill in indicators
        if is_end_of_week:
            for indicator in self.indicator_dict.values():
                indicator.push(df)

        for name, indicator in self.indicator_dict.items():
            df[name] = indicator.valueAt(0)

        return df

    def update_weekly_bar(self, i):
        bar = self.daily_df.iloc[i]
        if self.open is None: self.open = bar['Open']
        self.high = bar['High'] if self.high is None else max(self.high, bar['High'])
        self.low = bar['Low'] if self.low is None else min(self.low, bar['Low'])
        self.close = bar['Close']

    def convert(self):
        daily_count = self.daily_df.shape[0]
        for i in range(daily_count):
            today = self.daily_df.index[i]
            is_end_of_week = False
            if i < (daily_count - 1):
                tomorrow = self.daily_df.index[i+1]
                # if day of the week value of today is gte tomorrow
                # it indicates that tomorrow will be the start of a new week
                # therefore summarize and post the current weekly bar
                self.update_weekly_bar(i)
                if _weekday(today) >= _weekday(tomorrow):
                    is_end_of_week = True
                    self.last_week_bar = self._fill_bar(today, is_end_of_week)
                    # reset running weekly OHLC values
                    self.open, self.high, self.low, self.close = None, None, None, None
            else:
                if _weekday(today) == self.FRIDAY:
                    self.update_weekly_bar(i)
                    is_end_of_week = True
                    self.last_week_bar = self._fill_bar(today, is_end_of_week)

            self.last_week_bar['Date'] = today
            self.last_week_bar['EndOfWeek'] = is_end_of_week

            if self.weekly_df is None:
                self.weekly_df = pandas.DataFrame.copy(self.last_week_bar, deep=True)
            else:
                self.weekly_df = pandas.concat([self.weekly_df, self.last_week_bar], axis=0)


        self.weekly_df.set_index('Date', inplace = True)
        return self.weekly_df


class Mo(Indicator):
    def __init__(self, history_len, derived_len=50):
        super().__init__(history_len, derived_len)

    def _calculate(self):
        if len(self.history) >= self.history_len:
            n = self.history[-1]
            p = self.history[-self.history_len]
            m = n - p
            self.derived.append(m)

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
        super().__init__(history_len, derived_len)

    def _calculate(self):
        if len(self.history) >= self.history_len:
            m = list(self.history)[-self.history_len:]
            self.derived.append(statistics.mean(m))


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
            m = statistics.pstdev(self.history)
            self.derived.append(m)


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



def test_converter():

    df = create_simulated_timeseries(length=19)
    print(df)

    wk_df = WeeklyConverter(df).convert()
    print("\nconverted weekly\n")
    print(wk_df)


def test_converter_with_indicator():

    df = create_simulated_timeseries(length=19)
    print(df)

    wgt_close = WeightedClose(history_len=3)

    wk_df = WeeklyConverter(df, indicator_dict={'WgtClose':wgt_close}).convert()
    print("\nconverted weekly\n")
    print(wk_df)


def test_converter_with_ema():

    df = create_simulated_timeseries(length=19)
    close_ema = CloseEMA(coeff=0.3, warmup=2, history_len=50, derived_len=20)

    wk_df = WeeklyConverter(df, indicator_dict={'CloseEMA':close_ema}).convert()
    combined = df.join(wk_df, on='Date', rsuffix="_w")

    print(combined)


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
    #test_dataseries()
    #test_monday_anchor()
    #test_cross_dwn()
    #test_ema()
    #test_macd()
    #test_converter()
    #test_converter_with_indicator()
    #test_converter_with_ema()
    #test_volatility_stop()
    test_last_low()



