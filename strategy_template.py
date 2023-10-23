
import argparse
from datetime import date, datetime
import pandas
from collections import deque
from indicators import StDev, MondayAnchor 
import calendar_calcs
import pprint
from prettytable import PrettyTable  
import math

LONG = 'L'
SHORT = 'S'

class HighestValue(object):
    def __init__(self, v, history=50):
        self.highest = v
        self.history = history
        self.highs = deque()
        self.highs.append(v)

    def push(self, new_value):
        self.highest = max(self.highest, new_value)
        self.highs.append(self.highest)
        if len(self.highs) > self.history:
            self.highs.popleft()

class LowestValue(object):
    def __init__(self, v, history=50):
        self.lowest = v
        self.history = history
        self.lows = deque()
        self.lows.append(v)

    def push(self, new_value):
        self.lowest = min(self.lowest, new_value)
        self.lows.append(self.lowest)
        if len(self.lows) > self.history:
            self.lows.popleft()


def buy_stock(price, wallet, pct_alloc=1.0):
    shares = 0
    residual = wallet
    if wallet > 0 and price is not None:
        if price > 0:
            shares = int((pct_alloc * wallet)/price)
            residual = wallet - (shares * price)

    return shares, residual

def sell_stock(price, wallet, pct_alloc=1.0):
    shares = 0
    residual = wallet
    if wallet > 0 and price is not None:
        if price > 0:
            shares = int((pct_alloc * wallet)/price)
            residual = wallet - (shares * price)

    return -shares, residual


def mark_to_market(price, entry, position):
    trade_value = 0
    equity = 0 
    if entry is not None:
        trade_value = (price - entry) * position
        if position > 0:
            equity = position * price 
        else:
            equity = (abs(position) * entry) + trade_value 

    return equity, trade_value

def calculate_stats(trades, pnl_series):
    ## calc sharpe, drawdown, trade_count, etc
    dollars = [(x['Exit'] - x['Entry']) for x in trades]
    wins = [x for x in dollars if x > 0 ]

    trade_count = len(trades)
    win_pct = len(wins)/trade_count

    returns = (pnl_series['Equity'] / pnl_series['Equity'].shift(1)) - 1
    returns.dropna(inplace=True)

    dd, highest, lowest = 0, None, None
    for i in range(pnl_series.shape[0]):
        bar = pnl_series.iloc[i]
        v = bar['Equity']
        if highest is None or v > highest:
            highest = v
            lowest = v
        if v < lowest:
            lowest = v
        dd = max((highest/lowest)-1, dd)
    
    years = int(pnl_series.shape[0]/252.0)

    #total return
    totalRtn = (pnl_series.iloc[-1]['Equity']/pnl_series.iloc[0]['Equity']) - 1

    #compounded annualize growth rate
    cagr = ((pnl_series.iloc[-1]['Equity']/pnl_series.iloc[0]['Equity']) ** (1.0/years)) - 1
    sharpe = cagr/(returns.std() * math.sqrt(252))

    results = dict(Sharpe=sharpe,
                    Years=years,
                    CAGR=cagr,
                    TotalRtn=totalRtn,
                    Trades=trade_count,
                    WinPct=win_pct,
                    MaxDD=dd)

    return results


def adjust_prices(df):
    ## adjust the entire price bar to adjusted prices
    ## using the ratio of of Adj_Close/Close as multiplier
    data = []
    for i in range(df.shape[0]):
        bar = df.iloc[i]
        r = bar['Adj Close']/bar['Close']
        ah = bar['High'] * r
        al = bar['Low'] * r
        ao = bar['Open'] * r
        new_bar = [ao, ah, al, bar['Adj Close']]
        data.append([bar['Date']] + [round(x,6) for x in new_bar] + [bar['Volume']])

    nf = pandas.DataFrame(columns='Date Open High Low Close Volume'.split(), data=data)
    return nf, df
    

def calc_stop(side, anchor, volatility, multiplier, default=0.30):
    m = None
    if side == LONG:
        if volatility is not None:
            m = anchor - (volatility * multiplier)
        else:
            m = anchor * (1-default)

    if side == SHORT:
        if volatility is not None:
            m = anchor + (volatility * multiplier)
        else:
            m = anchor * (1+default)

    return m


def check_stop(position, price, stop_price):
    if position > 0 and price <= stop_price: 
        return True
    if position < 0 and price >= stop_price:
        return True

    return False


## modify this to handle rule-based exits according to position
## returns True/False if exit signal given, and a text string indicating type of exit
def check_exit(position, end_of_week, day_count, mtm):
    marker = '---'
    exit = False

    ## exit on first profitable day
    if mtm > 0:
        exit = True
        marker = 'PNL'
    elif day_count > 9:
        ## allow for trade to stay open for no more than 10 days
        exit = True
        marker = 'EXP'

    return exit, marker
        
## monday pullback strategy
## enter on close below monday's (first day of week) low.
## exit on first profitable day or after 10 days in duration
## standard volatility stop, no entries allowed on end_of_week 
def weekday_strategy(stock_name, wallet, start_from_dt=None, file_tag=None, use_raw=False):

    ## array of daily tuples:
    ## (Date Index_price Stock_Close TrendUp InDay Entry OutDay Position StopLevel MTM Equity)

    ## labels for file output
    if file_tag is None: 
        file_tag = '' 
    else:
        file_tag = f'.{file_tag}'

    sname = stock_name.upper()

    trade_series_columns = f'Date {sname}_Close TrendUp InDay Entry OutDay Position StopLevel MTM Equity'
    trade_series = []
    daily_table = PrettyTable(trade_series_columns.split())
    daily_table.float_format[f'{sname}_Close'] = ".6"
    daily_table.float_format['StopLevel'] = ".6"
    daily_table.float_format['MTM'] = ".2"
    daily_table.align['MTM'] = "r"
    daily_table.float_format['Equity'] = ".2"
    daily_table.align['Equity'] = "r"


    stock_file = f'/home/jcarter/sandbox/trading/data/{stock_name}.csv'
    print(f'Stock: {sname} -> {stock_file}')
    new_df, orig_df = adjust_prices(pandas.read_csv(stock_file))
    stock_df = orig_df if use_raw else new_df
    stock_df.set_index('Date',inplace=True)

    ### EXIT_CLOSE_ON_STOP = True/False flag to handle stop exits (either on close, or on next day's open)
    EXIT_CLOSE_ON_STOP = True 

    position = 0
    entry = None
    position_closed = None
    exit_on_open = False
    high_marker = None
    stop_level = None
    anchor_bar = None
    day_count = 0

    stdev = StDev(sample_size=50)
    anchor = MondayAnchor(derived_len=20)

    start_dt = None
    if start_from_dt is not None:
        start_dt = datetime.strptime(start_from_dt,"%Y-%m-%d").date()

    holidays = calendar_calcs.load_holidays()

    ## array of individual trade info:
    ## dict(Date,Position,Value,Duration,Entry,Exit)
    trades = []
    for i in range(stock_df.shape[0]):
        idate = stock_df.index[i]
        stock_bar = stock_df.loc[idate]

        cur_dt = datetime.strptime(idate,"%Y-%m-%d").date()

        anchor.push((cur_dt, stock_bar))

        cls = stock_bar['Close']
        stdev.push(cls)

        # collect all analytics, but don't start trading until we 
        # hit the the start_from_dt trading date
        if start_dt is not None and cur_dt < start_dt: continue

        end_of_week = calendar_calcs.is_end_of_week(cur_dt, holidays)

        buy_signal = False
        sell_signal = False
        if anchor.count() > 1:
            anchor_bar, bkout = anchor.valueAt(1)
            if bkout < 0 and end_of_week == False:
                buy_signal = True
       

        trend_marker =  "---"
        entry_marker = "---"
        exit_marker = "---"

        trade_mtm = 0
        equity = 0
        ### POSITION == 0: CHECK FOR NEW ENTRY
        if position == 0:
            day_count = 1
            entry = None
            if buy_signal:
                entry = stock_bar['Open']
                entry_marker = 'BUY'
                position, wallet = buy_stock(entry, wallet)
                high_marker = HighestValue(stock_bar['Open'])
                stop_level = calc_stop(LONG, high_marker.highest, stdev.valueAt(1), multiplier=2.5) 

            elif sell_signal:
                entry = stock_bar['Open']
                entry_marker = 'SELL'
                position, wallet = sell_stock(entry, wallet)
                low_marker = LowestValue(stock_bar['Open'])
                stop_level = calc_stop(SHORT, low_marker.lowest, stdev.valueAt(1), multiplier=2.5) 

            ## STOPPED OUT ON SAME BAR
            if position != 0:
                equity, trade_mtm = mark_to_market(stock_bar['Close'], entry, position)
                if check_stop(position, stock_bar['Close'], stop_level) == True:
                    exit_marker = 'STP'
                    if EXIT_CLOSE_ON_STOP:
                        trades.append(dict(InDate=idate,ExDate=idate,Position=position,Value=trade_mtm,Duration=1,Entry=entry,Exit=cls))
                        wallet += equity 
                        position_closed = True 
                        equity = 0
                    else:
                        exit_on_open = True

                ## EXIT ON SAME BAR
                else:
                    exit_triggered, exit_marker = check_exit(position, end_of_week, day_count, trade_mtm)
                    if exit_triggered:
                        trades.append(dict(InDate=idate,ExDate=idate,Position=position,Value=trade_mtm,Duration=1,Entry=entry,Exit=cls))
                        wallet += equity 
                        position_closed = True
                        equity = 0
                    else:
                        ## REGISTER A NEW TRADE 
                        exit_on_open = False
                        trades.append(dict(InDate=idate,ExDate='',Position=position,Value=0,Duration=0,Entry=entry,Exit=0))

        ### POSITION != 0: CHECK FOR EXITS
        else:
            day_count += 1
            if exit_on_open:
                equity, trade_mtm = mark_to_market(stock_bar['Open'], entry, position)
                current_trade = trades[-1]
                current_trade['Value'] = trade_mtm 
                current_trade['ExDate'] = cur_dt.strftime("%Y-%m-%d")
                trd_dt = datetime.strptime(current_trade['InDate'],"%Y-%m-%d").date()
                current_trade['Duration'] = ( cur_dt - trd_dt ).days
                current_trade['Exit'] = stock_bar['Open']

                ## add in liquidated position to wallet
                wallet += equity 
                position_closed = True
                equity = 0

                exit_on_open = False
            else:

                equity, trade_mtm = mark_to_market(stock_bar['Close'], entry, position)

                exit_triggered, exit_marker = check_exit(position, end_of_week, day_count, trade_mtm)
                stop_triggered = check_stop(position, stock_bar['Close'], stop_level)
                if stop_triggered: exit_marker = 'STP'

                if exit_triggered or (stop_triggered and EXIT_CLOSE_ON_STOP):
                    current_trade = trades[-1]
                    current_trade['Value'] = trade_mtm 
                    current_trade['ExDate'] = cur_dt.strftime("%Y-%m-%d")
                    trd_dt = datetime.strptime(current_trade['InDate'],"%Y-%m-%d").date()
                    current_trade['Duration'] = ( cur_dt - trd_dt ).days
                    current_trade['Exit'] = stock_bar['Close']

                    ## add in liquidated position to wallet
                    wallet += equity 
                    position_closed = True
                    equity = 0

                    exit_on_open = False
                else:
                    if not EXIT_CLOSE_ON_STOP and stop_triggered:
                        exit_on_open = True

        if position > 0:
            high_marker.push(stock_bar['High'])
            stop_level = max(stop_level, calc_stop(LONG, high_marker.highest, stdev.valueAt(1), multiplier=2.5))
        elif position < 0:
            low_marker.push(stock_bar['Low'])
            stop_level = min(stop_level, calc_stop(SHORT, low_marker.lowest, stdev.valueAt(1), multiplier=2.5))
        else:
            stop_level = None

        values = [idate, stock_bar['Close'], trend_marker, entry_marker, entry, exit_marker, position, stop_level, trade_mtm, (equity+wallet)]
        v = dict(zip(trade_series_columns.split(), values))
        trade_series.append(v)
        daily_table.add_row(values)

        if position_closed:
            position = 0
            position_closed = None


    tdf = pandas.DataFrame(data=trade_series)
    tdf = tdf[trade_series_columns.split()]
    pnl_series = tdf[['Date','Equity']]

    tdf.set_index('Date', inplace=True)
    pnl_series.set_index('Date', inplace=True)

    tdf.to_html(f'trade_series{file_tag}.html')
    tdf.to_csv(f'trade_series{file_tag}.csv')
    pnl_series.to_csv(f'pnl_series{file_tag}.csv')

    tts = pandas.DataFrame(data=trades)
    tts = tts['InDate ExDate Position Duration Entry Exit Value'.split()]
    tts['PNL'] = tts['Value'].cumsum()
    tts.set_index('InDate', inplace=True)
    tts.to_csv(f'trades{file_tag}.csv')

    print(" ")
    print(daily_table)
    print(" ")
    pprint.pprint(calculate_stats(trades, pnl_series))


if __name__ == '__main__':
    parser =  argparse.ArgumentParser()
    parser.add_argument("stock", help="stock to trade based on the index")
    parser.add_argument("--wallet", help="starting capital", type=int, default=10000)
    parser.add_argument("--start_from", help="simulation start date - format %Y-%m-%d")
    parser.add_argument("--tag", help="output file tag")
    parser.add_argument("--raw", help='use unadjusted prices flag', action='store_true')
    u = parser.parse_args()

    weekday_strategy(u.stock, u.wallet, u.start_from, u.tag, u.raw)


