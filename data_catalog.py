
import sys 
from datetime import datetime
import pandas
from prettytable import PrettyTable
from indicators import StDev

##
## data_table.py:  list the stats of the daily stock csv files 
## stored in the /data directory
##

def list_data(symbol_list):

    table_cols = "Count StartDt EndDt Symbol Close StDev Pct".split()
    daily_table = PrettyTable(table_cols)

    for col in "Count StartDt EndDt Symbol Close StDev Pct".split():
        daily_table.align[col] = "l"
        if col in ["Close", "StDev", "Pct"]:
            daily_table.float_format[col] = ".2"
            daily_table.align[col] = "r"

    for symbol in symbol_list:
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
        daily_table.add_row(values)

    print(" ")
    print(daily_table)
    print(" ")

if __name__ == '__main__':

    symbol_list = sys.argv[1:]
    list_data(symbol_list)
