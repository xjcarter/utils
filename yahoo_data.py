
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
import pandas
import sys
import argparse

## Grabs Daily Data from Yahoo Finance 
## Usage: python yahoo_data.py --list "AAA BBB" --file "symbolfile.txt" 

DATA_DIR = '/home/jcarter/sandbox/trading/data/'
START_DATE = '1999-12-31'

def get_daily_data(symbols):
   
    databank = dict()

    for sym in symbols:
    
        sym = sym.upper()
        sym = sym.replace('"','')
        if len(sym) == 0: continue
        filename = f'{DATA_DIR}/{sym}.csv'

        current_data = None
        last_date = START_DATE
        path = Path(filename)

        if path.is_file():
            # get last record
            current_data = pandas.read_csv(filename)
            last_row = current_data.shape[0]-1
            last_date = current_data.iloc[last_row]['Date']
            # re-index Date to a DateTimeIndex (which is the original form of the Yahoo download)
            current_data['Date'] =  pandas.to_datetime(current_data['Date'], format='%Y-%m-%d')
            current_data.set_index('Date', inplace=True)

        databank[sym] = current_data

        today = datetime.today()
        last_date = datetime.strptime(last_date,"%Y-%m-%d")

        days_to_fetch = today - last_date
        if days_to_fetch.days > 0:
            start = last_date +  timedelta(days=1)
            try:
                start, today = start.date(), today.date()
                print(f'Fetching: {sym}, {start}, {today}')
                new_data = yf.download(sym, datetime.strftime(start,"%Y-%m-%d"), datetime.strftime(today,"%Y-%m-%d"))
                if new_data is None:
                    print(f'Could not fetch data: {sym}, {start}, {today}')

                for col in ['Open','High','Low','Close','Adj Close']:
                    new_data[col] = new_data[col].round(decimals=5)

                if current_data is not None:
                    for ndate in new_data.index:
                        if ndate in current_data.index:
                            # remove duplicate row in currently saved data -> it will be updated with the new data
                            current_data.drop(ndate, inplace=True)
                    new_data = pandas.concat([current_data, new_data], axis=0)

                new_data.to_csv(filename)
                databank[sym] = new_data
            except:
                print(f'Could not process: {sym}, {start}, {today}')
                pass

    return databank

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
    parser.add_argument("--list", help="command line comma separated list of symbols", type=str, default="")
    parser.add_argument("--file", help="single entry per line symbol file", type=str, default="")
    u = parser.parse_args()
   
    symbols = parse_symbols(u.list, u.file)
    if len(symbols) > 0:
        get_daily_data(symbols)
        






        
