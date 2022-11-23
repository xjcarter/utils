
import argparse
import calendar
from datetime import date, datetime
import pandas
from collections import deque
from indicators import StDev
import statistics
import pprint
from prettytable import PrettyTable

##
## trade_tracker - simple utility that tracks a trade and provides stop and mtm info
## usage: python trade_tracker SYMBOL POSITION START_DT ENTRY_PRICE --end_dt EMD_DT --exit_prics EXIT_PRICE --csv
##
## --CSV flag to dump output to stdout as CSV
## --noheader: don't dump a header with csv file (for stitching trades together)
##
## example:
##  python trade_tracker.py UPRO 1000 2009-11-12 1.99 --end_dt 2009-12-11 --exit_price 3.0 --csv\
##


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


def mark_to_market(price, entry, position):
    trade_value = 0
    equity = position * price
    if entry is not None:
        trade_value = (price - entry) * position
    return equity, trade_value


def calc_stop(anchor, volatility, multiplier, default=0.30):
    m = None
    if volatility is not None:
        m = anchor - (volatility * multiplier)
    else:
        m = anchor * (1-default)
    return m


def track_trade(book_id, symbol, position, start_dt, entry_price, end_dt=None, exit_price=None, csv_out=False, no_header=False):

    stock_file = f'/home/jcarter/sandbox/trading/data/{symbol}.csv'
    stock_df = pandas.read_csv(stock_file)
    stock_df.set_index('Date',inplace=True)

    stdev = StDev(sample_size=50)

    start_dt = datetime.strptime(start_dt,"%Y-%m-%d").date()
    if end_dt is not None:
        end_dt = datetime.strptime(end_dt,"%Y-%m-%d").date()

    ## array of individual trade info:
    ## dict(Date,Position,Value,Duration,Entry,Exit)

    table_cols = "N Date Book Symbol Position Entry Close_Price Low_Price Stop Alert Next_Stop PNL MarketValue".split()
    daily_table = PrettyTable(table_cols)

    daily_table.float_format["PNL"] = ".2"
    daily_table.float_format["MarketValue"] = ".2"
    daily_table.align["MTM"] = "r"
    daily_table.align["Alert"] = "c"
    daily_table.align["Book"] = "c"
    for col in "Entry Close_Price Low_Price Stop Next_Stop".split():
        daily_table.float_format[col] = ".6"

    csv_table = list()
    if no_header == False: csv_table.append(",".join(table_cols))

    high_marker = None
    stop_level = None
    n = 0
    for i in range(stock_df.shape[0]):
        idate = stock_df.index[i]
        stock_bar = stock_df.loc[idate]

        cur_dt = datetime.strptime(idate,"%Y-%m-%d").date()

        close_price = stock_bar['Close']
        stdev.push(close_price)

        # collect all analytics, but don't start trading until we
        # hit the the start trading date
        if start_dt is not None and cur_dt < start_dt: continue

        alert = "-------"
        if cur_dt == start_dt:
            high_marker = HighestValue(entry_price)
            # set first stop off of entry price
            stop_level = calc_stop(high_marker.highest, stdev.valueAt(1), multiplier=2.5)
            alert = "ENTRY"

        if stock_bar['Close'] <= stop_level: alert = 'OUT'

        mark_price = stock_bar['Close']
        low_price = stock_bar['Low']

        if end_dt is not None and cur_dt == end_dt:
            if exit_price is not None: mark_price = exit_price
            alert = 'EXIT'

        equity, trade_mtm = mark_to_market(mark_price, entry_price, position)

        high_marker.push(stock_bar['High'])

        #calc stop level for tomorrow
        next_stop = max(stop_level, calc_stop(high_marker.highest, stdev.valueAt(1), multiplier=2.5))

        ## duration counter
        n += 1

        values = [n, idate, book_id, symbol, position, entry_price, mark_price, low_price, stop_level, alert, next_stop, trade_mtm, equity]
        csv_table.append(",".join([str(x) for x in values]))
        daily_table.add_row(values)

        #update stop level 
        stop_level = next_stop

        if end_dt is not None and cur_dt == end_dt: break

    if csv_out:
        for row in csv_table: print(row)
    else:
        print(" ")
        print(daily_table)
        print(" ")


if __name__ == '__main__':

    parser =  argparse.ArgumentParser()
    parser.add_argument("symbol", help="tracking asset")
    parser.add_argument("position", help="position in shares", type=int)
    parser.add_argument("start_dt", help="starting trade date")
    parser.add_argument("entry_price", help="entry price", type=float)
    parser.add_argument("--end_dt", help="ending trade date", default=None)
    parser.add_argument("--exit_price", help='exit price', type=float, default=None)
    parser.add_argument("--csv", help='dump as CSV file', action='store_true')
    parser.add_argument("--noheader", help='dont include header with CSV file', action='store_true')
    parser.add_argument("--book", help="book label", default="")
    u = parser.parse_args()

    track_trade(u.book, u.symbol, u.position, u.start_dt, float(u.entry_price), u.end_dt, u.exit_price, u.csv, u.noheader)
