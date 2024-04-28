from indicators import LogRtn, indicator_to_df
import pandas

## test the indicator_to_df function
## indicator_to_df take price bars, an Indicator object, and returns a DataFrame
## of that indicator 
## 
## the 'attach_transform' picks from the price_series DataFrame provided
## to provide the correct input to the Indicator object in question
##
## definitely will come in handy!
##

def use_close(bar):
    return bar['Close']

df = pandas.read_csv('/home/jcarter/work/trading/data/SPY.csv')

## create a 5-dayMA using the close 
wk = LogRtn(5)
mth = LogRtn(20)
yr = LogRtn(240)

wk.attach_transform(use_close)
mth.attach_transform(use_close)
yr.attach_transform(use_close)

idf = indicator_to_df(df,wk,'wkRtn', merge=True)
idf = indicator_to_df(idf,mth,'mthRtn', merge=True)
idf = indicator_to_df(idf,yr,'yrRtn', merge=True)

xdf = idf[['Date','wkRtn','mthRtn','yrRtn']]

wks = idf[~idf['wkRtn'].isna()]['wkRtn']
mths = idf[~idf['mthRtn'].isna()]['mthRtn']
yrs = idf[~idf['yrRtn'].isna()]['yrRtn']

pos_wks = wks[wks > 0]
pos_mths = mths[mths > 0]
pos_yrs = yrs[yrs > 0]

print('WIN RATES:')
for label, pos, tot in [ ('WEEKLY', pos_wks, wks), ( 'MONTHLY', pos_mths, mths), ( 'YEARLY', pos_yrs, yrs) ]:
    v = len(pos)
    b = len(tot)
    q = v/b
    print(f'{label}: win% = {v/b:.2f}, wins = {v}, total= {b}') 

for  g, lbl in [(wks, 'WK'), (mths, 'MTH'), (yrs, 'YR')]:
    print(f'\n{lbl}')
    print(g.describe())

