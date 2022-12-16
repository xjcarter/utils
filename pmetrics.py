
import pandas as pd
import numpy as np

def calculate_return_series(price_series, use_log_rtns=False):
    rtns = None
    if use_log_rtns:
        rtns = np.log(price_series/price_series.shift(1, axis=0))
    else:
        rtns = (price_series/prices.shift(1, axis=0)) - 1 
    rtns.dropna(inplace=True)
    return rtns

def get_years_past(tseries):
    return tseries.shape[0]/ 252.0 

def calc_annualized_volatility(return_series):
    return return_series.std() * np.sqrt(252.0)

## calc Compound Annualized Growth Rate
def calculate_cagr(tseries):
    years_past = get_years_past(tseries)
    total_rtn = tseries.iloc[-1] / tseries.iloc[0]
    cagr = (total_rtn ** (1/years_past)) - 1

def calculate_sharpe(price_series):
    cagr = calculate_cagr(price_series)
    rtns = calculate_return_series(price_series, use_log_rtns=True)
    vol = calculate_annualized_volatility(rtns)
    return cagr/vol

