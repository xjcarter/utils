
import sys 
from datetime import datetime
import pandas
from prettytable import PrettyTable
from indicators import StDev
import argparse

##
## data_table.py:  list the stats of the daily stock csv files 
## stored in the /data directory
##

def list_data(symbol_list, count_limit, names_only):

    table_cols = "Count StartDt EndDt Symbol Close StDev Pct".split()
    daily_table = PrettyTable(table_cols)

    for col in "Count StartDt EndDt Symbol Close StDev Pct".split():
        daily_table.align[col] = "l"
        if col in ["Close", "StDev", "Pct"]:
            daily_table.float_format[col] = ".2"
            daily_table.align[col] = "r"

    errors = []
    for symbol in symbol_list:
        symbol = symbol.replace('"','')
        if len(symbol) == 0: continue
        try:
            stock_file = f'/home/jcarter/sandbox/trading/data/{symbol}.csv'
            stock_df = pandas.read_csv(stock_file)
            stock_df.set_index('Date', inplace=True)

            stdev = StDev(sample_size=50)

            count = stock_df.shape[0]
            start_dt = datetime.strptime(stock_df.index[0],"%Y-%m-%d").date()
            end_dt = datetime.strptime(stock_df.index[-1],"%Y-%m-%d").date()
            close_price = stock_df.loc[stock_df.index[-1]]['Close'] 

            for i in range(count-60, count):
                idate = stock_df.index[i]
                stock_bar = stock_df.loc[idate]
                close_price = stock_bar['Close']
                stdev.push(close_price)

            q = stdev.valueAt(0)
            values = [count, start_dt, end_dt, symbol, close_price, q, q/close_price]

            if count >= count_limit:
                ## create a data table, otherwise just list the filtered names
                if not names_only:
                    daily_table.add_row(values)
                else:
                    print(symbol)
        except: 
           errors.append(f'Cant find: {stock_file}')  

    if not names_only:
        print(" ")
        print(daily_table)
        print(" ")

    for y in errors:
        print(y)

def parse_symbols(sym_string, sym_file):
    symbols = []
    if sym_string is not None and len(sym_string) > 0:
        symbols = sym_string.split()

    file_symbols = []
    if sym_file is not None and len(sym_file) > 0:
        with open(sym_file, 'r') as f:
            file_symbols = f.readlines()

    ## clean whitespace
    v = [x.strip() for x in symbols + file_symbols]
    return v


if __name__ == '__main__':
    parser =  argparse.ArgumentParser()
    parser.add_argument("--list", help="command line SPACE separated list of symbols", type=str, default="")
    parser.add_argument("--file", help="single entry per line symbol file", type=str, default="")
    parser.add_argument("--count_limit", help="limit of dataset size", type=int, default=0)
    parser.add_argument("--names_only", help="just list symbol names, no data table", action='store_true')
    u = parser.parse_args()

    symbol_list = parse_symbols(u.list, u.file) 
    list_data(symbol_list, u.count_limit, u.names_only)
