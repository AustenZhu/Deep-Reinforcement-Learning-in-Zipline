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

import numpy as np
from zipdl.data import db
from zipdl.data import models as m
session = db.create_session()

import datetime as dt
import pandas as pd

METRIC_DICT = {}

#==========Ticker Fundamental Utilities=========================================
memoized = {}
def get_current_universe(date):
    if date in memoized:
        return memoized[date]
    all_stocks = session.query(m.Fundamentals).filter(m.Fundamentals.metric == 'Gross Margin').all()
    universe = []
    try:
        curr = dt.datetime.strptime(date, '%Y-%m-%d')
    else:
        assert isinstance(curr, dt.datetime)
    for stock in all_stocks:
        series = pd.Series(stock.time_series)
        first = dt.datetime.strptime(series.index[0], '%Y-%m-%d')
        if first <= curr:
            universe.append(stock.ticker)
    memoized[date] = universe
    return universe


#==========Market Metric Utilities===============================================

def translate_metric(value, bins):
    for num, b in enumerate(bins):
        if value > b: 
            return num
    else: 
        return len(bins)

def get_metric(date, metric, session=db.create_session()):
    assert metric in [x.metric for x in session.query(Market_Metric).all()]
    metric = session.query(m.Market_Metric).filter(m.Market_Metric.metric==metric).all()[0]
    series = metric.time_series
    strings_to_date(series, '%Y-%m-%d')
    closest_date = find_closest_date(series.keys(), date)
    return series[closest_date]

def get_metric_bucket(date, metric):
    assert isinstance(date, dt.datetime)
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

def find_closest_date(items, pivot):
    return min(items, key=lambda x: abs((x - pivot).days))

def calc_sortino(perf):
    returns = perf['portfolio_value'][::5].pct_change()[1:]
    weekly_mar = 0.1 ** (1/52) #10% annualized minimum accepted return
    #Target downside deviation, as calculated here: https://www.sunrisecapital.com/wp-content/uploads/2013/02/Futures_Mag_Sortino_0213.pdf
    tdd = np.sqrt(np.sum([min(0, returns - weekly_mar)]))
    return (returns.mean() - weekly_mar)/tdd

def strings_to_date(mydict, str_format):
    origlen = len(mydict.keys())
    for key in list(mydict.keys()):
        mydict[datetime.strptime(key, str_format)] = mydict[key]
        del mydict[key]
    assert origlen == len(mydict.keys())
def dates_to_string(mydict):
    origlen= len(mydict.keys())
    for key in list(mydict.keys()):
        if type(key) is not str:
            try:
                mydict[date.strftime(key, '%Y-%m-%d')] = mydict[key]
            except:
                try:
                    mydict[str(key)] = mydict[key]
                except:
                    pass
            del mydict[key]
    assert origlen == len(mydict.keys())