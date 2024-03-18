
import argparse
from datetime import datetime
import pandas
from indicators import WeeklyBar
import os

DATA_DIR = os.getenv('DATA_DIR', '/home/jcarter/work/trading/data/')

## convert a daily bar time series to weekly bar time series ...

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


def convert_to_weekly_data(stock, use_raw, newfile=None):
    stock_file = f'{DATA_DIR}/{stock}.csv'
    new_df, orig_df = adjust_prices(pandas.read_csv(stock_file))
    stock_df = new_df
    if use_raw: stock_df = orig_df

    wk_converter = WeeklyBar()
    wk_df = wk_converter.convert_daily_dataframe(stock_df)

    outfile = f'{DATA_DIR}/{stock}.w.csv'
    if newfile: outfile =newfile
    print(f'writing weekly data file: {outfile}')
    wk_df.to_csv(f'{outfile}', index=False)

if __name__ == '__main__':

    parser =  argparse.ArgumentParser()
    parser.add_argument("stock", help="stock to track")
    parser.add_argument("--raw", help="show raw data", action='store_true')
    parser.add_argument("--output", help="show raw data", default=None)
    u = parser.parse_args()

    convert_to_weekly_data(u.stock, u.raw, u.output)

