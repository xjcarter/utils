
import pandas
import calendar
from datetime import date, datetime, timedelta

MONDAY = 0
TUESDAY = 1 
WEDNESDAY = 2 
THURSDAY = 3 
FRIDAY = 4
SATURDAY = 5
SUNDAY = 6
WEEKDAYS = ['MON','TUE','WED','THU','FRI','SAT','SUN']

def cvt_date(str):
    return datetime.strptime(str,"%Y-%m-%d").date()

def weekday(dt):
    return cvt_date(dt).weekday()

def business_days(trade_date):
    cal = calendar.Calendar()
    month_cal = cal.monthdatescalendar(trade_date.year, trade_date.month)
    busi_days = []
    for week in month_cal:
        for d in week:
            if d.weekday() < SATURDAY and d.month == trade_date.month:
                busi_days.append(d)

    return busi_days

## given a date, 
## return the index of that date in the month of business days
def day_of_month(dt, holidays):
    v = dt
    if isinstance(dt, str): v = cvt_date(dt)
    mth = []
    for d  in business_days(v):
        if d not in holidays: mth.append(d)
    if v in mth: return mth.index(v) + 1
    return None

## given an index
## return the date in the month at that index of business days
def day_in_month(mth, year, day_in_month, holidays):
    v = cvt_date(f'{year}-{mth:02d}-01') 
    mth = []
    for d  in business_days(v):
        if d not in holidays: mth.append(d)
    if mth:
        assert(day_in_month != 0)
        if day_in_month > 0: day_in_month -= 1
        return mth[day_in_month]
    return None

def last_day_of_month(cur_dt, holidays):
    last_dt = day_in_month(cur_dt.month, cur_dt.year, -1, holidays)
    return cur_dt == last_dt

def first_day_of_month(cur_dt, holidays):
    first_dt = day_in_month(cur_dt.month, cur_dt.year, 1, holidays)
    return cur_dt == first_dt

## return cardinal week value for given date
## ie. 2022-02-3 is in the 1st week of the month
## therefore, this func would return 1
## if negate last week flag is set, the function
## will return the last week of the month as a negative value: i.e -4 or -5
def week_in_month(dt, holidays, negate_last_week=False):
    v = dt
    if isinstance(dt, str): v = cvt_date(dt)
    mth = []
    for d  in business_days(v):
        if d not in holidays: mth.append(d)
    if mth:
        weeks=[1] * len(mth)
        for i in range(1,len(mth)):
            increment = 0
            prev = mth[i-1].weekday()
            today = mth[i].weekday()
            if prev > today: increment = 1 
            weeks[i] = weeks[i-1] + increment
        if v in mth:
            weeks_in_mth = max(weeks)
            w = weeks[mth.index(v)] 
            ## flag last week of the month with a negative value
            if negate_last_week and w == weeks_in_mth:
                return -w
            return w

    return None
    

## find the first trading day of the week 
def is_start_of_week(trade_date, holidays):
    firsts = []
    find_next_day = False
    for d in business_days(trade_date):
        if find_next_day == False:
            if d.weekday() == MONDAY:
                if d not in holidays:
                    firsts.append(d) 
                else:
                    find_next_day = True
        elif d.weekday() < SATURDAY and d not in holidays:
            firsts.append(d) 
            find_next_day = False

    return trade_date in firsts


def is_day_of_week(tgt_day, trade_date, holidays):
    firsts = []
    find_next_day = False
    for d in business_days(trade_date):
        if find_next_day == False:
            if d.weekday() == tgt_day:
                if d not in holidays:
                    firsts.append(d) 
                else:
                    find_next_day = True
        elif d.weekday() < SATURDAY and d not in holidays:
            firsts.append(d) 
            find_next_day = False

    return trade_date in firsts


## find previous trading day relative to the date given
def prev_trading_day(reference_date, holidays):
    one_day_back = timedelta(days=1)
    d = reference_date - one_day_back
    while True:
        if d.weekday() < SATURDAY and d.strftime("%Y-%m-%d") not in holidays:
            return date(d.year, d.month, d.day)
        else:
            d -= one_day_back


## find the last trading day of the week 
def is_end_of_week(trade_date, holidays):
    ends = [] 
    find_prev_day = False
    for d in business_days(trade_date)[::-1]:
        if find_prev_day == False:
            if d.weekday() == FRIDAY:
                if d not in holidays:
                    ends.append(d)
                else:
                    find_prev_day = True
        elif d.weekday() < SATURDAY and d not in holidays:
            ends.append(d)
            find_prev_day = False

    return trade_date in ends


def load_holidays():
    ## file foramt: "Date,Holiday"
    fn = "/home/jcarter/work/trading/data/us_market_holidays.csv"
    df = pandas.read_csv(fn)
    holidays = []
    for d in df['Date'].to_list():
        holidays.append(datetime.strptime(d,"%Y-%m-%d").date())
    return holidays

def show_trading_calendar(year, month):
    holidays = load_holidays()
    cal_columns = f'Date Action Day Holiday'
    cal_table = PrettyTable(cal_columns.split())
    cal = calendar.Calendar()
    test_month = cal.monthdatescalendar(year, month)
    #print('holidays: ', holidays)
    for week in test_month:
        for d in week:
            if d.month == month:
                hh = "-----" if d not in holidays else "HOLIDAY" 
                action = "----"
                if check_buy(d,holidays):
                    action = "BUY"
                elif check_sell(d,holidays):
                    action = "SELL"
                #print(d, action, WEEKDAYS[d.weekday()], hh)
                cal_table.add_row([d, action, WEEKDAYS[d.weekday()], hh])
    
    print(" ")
    print(cal_table)
   

if __name__ == '__main__':
    g = cvt_date("2022-08-16")
    print(business_days(g))

    print(WEEKDAYS[weekday("2022-08-16")])

