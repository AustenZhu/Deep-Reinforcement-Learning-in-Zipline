import numpy as np
import pandas as pd
import pandas_datareader as pdr
import fix_yahoo_finance as yf

from zipdl.data import db
from zipdl.data import models
from datetime import datetime, date
import datetime as dt
yf.pdr_override()

session = db.create_session()

#==================UTILS===================
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
#===========================================
#VIX
vix = pd.read_csv('vixcurrent.csv')
vix = vix.set_index('Date')
vixD = vix['VIX Close'].to_dict()
strings_to_date(vixD, '%m/%d/%Y')
dates_to_string(vixD)
mm = models.Market_Metric(metric='vix', time_series=vixD)
session.add(mm)

#Vix deltas
vixC = np.array(vix['VIX Close'])
vix1w = (vixC[:-5] - vixC[5:])/vixC[:-5]
vix1w = pd.Series(vix1w) 
vix1w.index = vix.index[:-5]
vix1m = (vixC[:-20] - vixC[20:])/vixC[:-20]
vix1m = pd.Series(vix1m) 
vix1m.index = vix.index[:-20]
vix2w = (vixC[:-10] - vixC[10:]) / vixC[:-10]
vix2w = pd.Series(vix2w) 
vix2w.index = vix.index[:-10]
mm = models.Market_Metric(metric='vixD1w', time_series=vix1w.to_dict())
session.add(mm)
mm = models.Market_Metric(metric='vixD1m', time_series=vix1m.to_dict())
session.add(mm)
mm = models.Market_Metric(metric='vixD2w', time_series=vix2w.to_dict())
session.add(mm)

#Personal Savings:
df = pd.read_csv('personal_savings.csv').set_index('DATE')
df = df['PSAVERT']
nums = np.array(df)

mm = models.Market_Metric(metric='personal savings', time_series=df.to_dict())
session.add(mm)
difference = (nums[:-1] - nums[1:]) / nums[:-1]
series = pd.Series(difference.squeeze())
series.index = df.index[:-1]
mm = models.Market_Metric(metric='ps-1mdelat', time_series=series.to_dict())
session.add(mm)

#SPY TTM Returns and variants
start = '2002-01-01'
spy = pdr.get_data_yahoo('SPY', start=start)
spy = spy['Close']
start = datetime.strptime(start, '%Y-%m-%d')

def t_some_m(months):
    days = 31 * months
    shift = dt.timedelta(days=days) - dt.timedelta(days = 1)
    shift_start = start + shift
    try:
        index = date.strftime(shift_start, '%Y-%m-%d')
        shift = spy.loc[index:]
    except: 
        shift_start = shift_start + date.timedelta(days=2)
        shift = spy.loc[index:]
    used_spy = spy.iloc[:len(shift)]
    returns = (np.array(shift) - np.array(used_spy)) / np.array(used_spy)
    returns = pd.Series(returns)
    returns.index = shift.index
    return returns

#Calc t3m
returns = t_some_m(3).to_dict()
dates_to_string(returns)
mm = models.Market_Metric(metric='t3m', time_series=returns)
session.add(mm)

#Calc t6m
returns1 = t_some_m(6).to_dict()
dates_to_string(returns1)
mm = models.Market_Metric(metric='t6m', time_series=returns1)
session.add(mm)

#Calc ttm
returns2 = t_some_m(12).to_dict()
dates_to_string(returns2)
mm = models.Market_Metric(metric='ttm', time_series=returns2)
session.add(mm)

session.commit()




