
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
import pandas
import sys

## Grabs Daily Data from Yahoo Finance 
## Usage: python yahoo_data.py ^DJI ^RUT IBM GOOG

DATA_DIR = '/home/jcarter/sandbox/trading/data/'
START_DATE = '1999-12-31'

def get_daily_data(symbols):
   
    databank = dict()

    for sym in symbols:
    
        sym = sym.upper()
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
                for col in ['Open','High','Low','Close','Adj Close']:
                    new_data[col] = new_data[col].round(decimals=5)
                if current_data is not None:
                    new_data = pandas.concat([current_data, new_data], axis=0)

                new_data.to_csv(filename)
                databank[sym] = new_data
            except:
                print(f'Could not fetch data for: {sym}, {start}, {today}')
                pass

    return databank


if __name__ == '__main__':
    
    symbols = sys.argv[1:]
    if len(symbols) > 0:
        get_daily_data(symbols)
        






        
