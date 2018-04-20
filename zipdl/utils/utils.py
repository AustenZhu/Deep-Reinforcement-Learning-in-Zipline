'''
TODO: Grab metric from db

#Current, untested methodology for creating buckets - look at the 33th and 66th percentiles.
'''
VIX_BUCKETS = [13.71, 18.4386]
VIXD1W_BUCKETS = [-0.0401454053566, 0.051352106388]
VIXD2W_BUCKETS = [-0.0511173667494, 0.0638941066051]
VIXD1M_BUCKETS = [-0.0611542216514, 0.086399260223]
T3M_BUCKETS  = [0.00155393602565, 0.0476175793035]
T6M_BUCKETS = [0.0193284320648, 0.0806490876864]
TTM_BUCKETS = [0.057386937542, 0.145046716341]
SAVINGSD1M_BUCKETS = [-0.0261929824561, 0.0357532467532]
SAVINGS_BUCKETS = [4.5, 5.6]
FUNDAMENTALS = ['Gross Margin', 'Cash From Investing Activities', 'Minorities', 'Free Cash Flow', 'Total Noncurrent Assets', 'Net PP&E', 'Income Taxes', 'Depreciation & Amortisation', 'Change in Working Capital', 'Equity Before Minorities', 'Cash and Cash Equivalents', 'SG&A', 'Goodwill', 'Cash From Operating Activities', 'Return on Equity', 'Retained Earnings', 'Current Assets', 'Net Change in PP&E & Intangibles', 'Dividends', 'Accounts Payable', 'Return on Assets', 'Net Change in Cash', 'Current Ratio', 'Net Income from Discontinued Op.', 'Treasury Stock', 'Total Noncurrent Liabilities', 'Share Capital', 'Revenues', 'Long Term Debt', 'EBITDA', 'Total Assets', 'Total Liabilities', 'Intangible Assets', 'EBIT', 'Net Profit Margin', 'R&D', 'COGS', 'Liabilities to Equity Ratio', 'Abnormal Gains/Losses', 'Net Profit', 'Preferred Equity', 'Total Equity', 'Debt to Assets Ratio', 'Current Liabilities', 'Receivables', 'Operating Margin', 'Short term debt', 'Cash From Financing Activities', 'Shares_Outstanding', 'VALUE']


import numpy as np
from zipdl.data import db
from zipdl.data import models as m
import datetime as dt
from dateutil import parser
import pandas as pd

METRIC_DICT = {}
session=db.create_session(autoflush=False)
#==========Ticker Fundamental Utilities=========================================
memoized = {}
def get_current_universe(date):
    if date in memoized:
        return memoized[date]
    all_stocks = session.query(m.Fundamentals).filter(m.Fundamentals.metric == 'EBITDA').all()
    universe = []
    try:
        date = date.replace(tzinfo=None)
    except:
        pass
    try:
        curr = dt.datetime.strptime(date, '%Y-%m-%d')
    except:
        assert isinstance(date, dt.datetime)
        curr = date
    for stock in all_stocks:
        series = pd.Series(stock.time_series)
        try:
            first = dt.datetime.strptime(series.index[0], '%Y-%m-%d')
        except:
            first = series.index[0]
        if first <= curr:
            universe.append(stock.ticker)
    memoized[date] = universe
    return universe

def find_closest_date(items, pivot):
    pivot = pivot.replace(tzinfo=None)
    return min(items, key=lambda x: abs((x - pivot).days))

#==========Market Metric Utilities===============================================

def translate_metric(value, bins):
    for num, b in enumerate(bins):
        if value < b: 
            return num
    else: 
        return len(bins)

def get_metric(date, metric, session=session):
    metric = session.query(m.Market_Metric).filter(m.Market_Metric.metric==metric).one_or_none()
    series = metric.time_series
    series = safe_strings_to_date(series)
    if not isinstance(date, dt.datetime):
        date = dt.datetime.strptime(date, '%Y-%m-%d')
    closest_date = find_closest_date(series.keys(), date)
    return series[closest_date]

def get_metric_bucket(date, metric):
    if not isinstance(date, dt.datetime):
        date = dt.datetime.strptime(date, '%Y-%m-%d')
    assert metric in list(METRIC_DICT.keys())
    val = get_metric(date, metric)
    return METRIC_DICT[metric](val)

def transform_vix(value):
    return translate_metric(value, VIX_BUCKETS)
METRIC_DICT['vix'] = transform_vix

def transform_vixD1w(value):
    return translate_metric(value, VIXD1W_BUCKETS)
METRIC_DICT['vixD1w'] = transform_vixD1w

def transform_vixD2w(value):
    return translate_metric(value, VIXD2W_BUCKETS)
METRIC_DICT['vixD2w'] = transform_vixD2w

def transform_vixD1m(value):
    return translate_metric(value, VIXD1M_BUCKETS)
METRIC_DICT['vixD1m'] = transform_vixD1m

def transform_savings(value):
    return translate_metric(value, SAVINGS_BUCKETS)
METRIC_DICT['personal savings'] = transform_savings

def transform_savingsD1m(value):
    return translate_metric(value, SAVINGSD1M_BUCKETS)
METRIC_DICT['ps-1mdelat'] = transform_savingsD1m

def transform_t3m(value):
    return translate_metric(value, T3M_BUCKETS)
METRIC_DICT['t3m'] = transform_t3m

def transform_t6m(value):
    return translate_metric(value, T6M_BUCKETS)
METRIC_DICT['t6m'] = transform_t6m

def transform_ttm(value):
    return translate_metric(value, TTM_BUCKETS)
METRIC_DICT['ttm'] = transform_ttm

#===================Fundamentals=============================
#MAKE SURE TO CALL CLEAN_DB_TIME_SERIES BEFORE ACCESSING FUNDAMENTALS
def get_fundamental(date, fundamental, ticker, session=session):
    assert fundamental in FUNDAMENTALS
    #print(ticker)
    data = session.query(m.Fundamentals).filter(m.Fundamentals.metric==fundamental).filter(m.Fundamentals.ticker==ticker).one_or_none()
    if not data:
        return np.nan
    series = data.time_series
    closest_date = find_closest_date(series.keys(), date)
    return series[closest_date]

from zipdl.data import models as m
def get_fundamentals(date, fundamental, tickers, session=session):
    data = session.query(m.Fundamentals).filter(m.Fundamentals.metric==fundamental).all()
    data = [obj for obj in data if obj.ticker in tickers]
    dict_tickers = [obj.ticker for obj in data]
    data.sort(key = lambda obj: tickers.index(obj.ticker))
    def get_close(obj, date):
        #print(obj.ticker)
        if not obj.time_series:
            return np.nan
        close = utils.find_closest_date(obj.time_series.keys(), date)
        if (close - date) > 7: 
            return np.nan
        return obj.time_series[close]
    values = [get_close(obj, date) for obj in data]
    dictionary = dict(zip(dict_tickers, values))
    def ret_correct(ticker):
        if ticker in dict_tickers:
            index = dict_tickers.index(ticker)
            return values[index]
        return np.nan
    return [ret_correct(ticker) for ticker in tickers]

def calc_sortino(perf):
    returns = perf['portfolio_value'][::5].pct_change()[1:]
    weekly_mar = 0.1 ** (1/52) #10% annualized minimum accepted return
    #Target downside deviation, as calculated here: https://www.sunrisecapital.com/wp-content/uploads/2013/02/Futures_Mag_Sortino_0213.pdf
    tdd = np.sqrt(np.sum([min(0, returns - weekly_mar)]))
    return (returns.mean() - weekly_mar)/tdd

def clean_db_time_series(fundamentals, session=session):
    for fundamental in fundamentals:
        print('cleaning {}'.format(fundamental))
        objs = session.query(m.Fundamentals).filter(m.Fundamentals.metric==fundamental).all()
        if fundamental == 'Shares_Outstanding':
            for obj in objs:
                strings_to_date(obj.time_series, wierd=True)
        else:
            for obj in objs:
                strings_to_date(obj.time_series)

def strings_to_date(mydict, wierd=False):
    origlen = len(mydict.keys())
    for key in list(mydict.keys()):
        if not wierd:
            try:
                mydict[parser.parse(key)] = mydict[key]
                del mydict[key]
            except TypeError:
                pass
        else:
            try:
                date = dt.datetime.strptime(key, '%m/%d/%Y')
                mydict[date] = mydict[key]
                del mydict[key]
            except TypeError:
                pass
            except:
                del mydict[key] 
def dates_to_string(mydict):
    origlen= len(mydict.keys())
    for key in list(mydict.keys()):
        if type(key) is not str:
            try:
                mydict[dt.datetime.strftime(key, '%Y-%m-%d')] = mydict[key]
            except:
                pass
            del mydict[key]
    assert origlen == len(mydict.keys())

def safe_strings_to_date(mydict, wierd=False):
    mydict = mydict.copy()
    origlen = len(mydict.keys())
    for key in list(mydict.keys()):
        if not wierd:
            try:
                mydict[parser.parse(key)] = mydict[key]
                del mydict[key]
            except TypeError:
                pass
        else:
            try:
                date = dt.datetime.strptime(key, '%m/%d/%Y')
                mydict[date] = mydict[key]
                del mydict[key]
            except TypeError:
                pass
            except:
                del mydict[key]
    return mydict 

def reload_session():
    session = db.create_session(autoflush=False)