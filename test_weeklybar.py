from indicators import WeeklyBar
import pandas

DATA_DIR = '/home/jcarter/work/trading/data/'

wb = WeeklyBar()

infile = f'{DATA_DIR}/SPY.csv'

df = wb.convert_daily_file(infile)

print(df.tail(20))
