from indicators import MA, indicator_to_df
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
ma5 = MA(5)
ma5.attach_transform(use_close)

idf = indicator_to_df(df,ma5,'ma5', merge=True)

print(idf.head(10))

"""
sample output:
         Date       Open       High        Low      Close  Adj Close      Volume        ma5
0  2000-01-03  148.25000  148.25000  143.87500  145.43750   96.55506   8164300.0        NaN
1  2000-01-04  143.53125  144.06250  139.64062  139.75000   92.77916   8089800.0        NaN
2  2000-01-05  139.93750  141.53125  137.25000  140.00000   92.94514  12177900.0        NaN
3  2000-01-06  139.62500  141.50000  137.75000  137.75000   91.45140   6227200.0        NaN
4  2000-01-07  140.31250  145.75000  140.06250  145.75000   96.76252   8066500.0  141.73750
5  2000-01-10  146.25000  146.90625  145.03125  146.25000   97.09446   5741700.0  141.90000
6  2000-01-11  145.81250  146.09375  143.50000  144.50000   95.93265   7503700.0  142.85000
7  2000-01-12  144.59375  144.59375  142.87500  143.06250   94.97834   6907700.0  143.46250
8  2000-01-13  144.46875  145.75000  143.28125  145.00000   96.26462   5158300.0  144.91250
9  2000-01-14  146.53125  147.46875  145.96875  146.96875   97.57167   7437300.0  145.15625
"""

