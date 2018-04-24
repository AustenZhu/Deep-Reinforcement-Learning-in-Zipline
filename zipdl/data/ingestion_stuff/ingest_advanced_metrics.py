import pandas as pd
import numpy as np  
from zipdl.data import db
from zipdl.data import models as m
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
yf.pdr_override()
import quandl
import os
quandl.ApiConfig.api_key = os.environ['QUANDL-APICONFIG']
from zipdl.utils import utils

UNIVERSE = pd.read_csv('condensed_universe.csv')
session = db.create_session(autoflush=False)

def safe_pdr_get(ticker, start=None, end=None):
    tries = 0
    data = quandl.get_table('WIKI/PRICES', qopts = { 'columns': ['date', 'close'] }, ticker = ticker, date = { 'gte': start, 'lte': end})
    data = data.set_index('date')['close']
    return data

no_shares = 0
for ticker in UNIVERSE:
    ticker = ticker.replace("'", "").replace(' ', '')
    print('Ingesting {}'.format(ticker))
    shares_outstanding = session.query(m.Fundamentals).filter(m.Fundamentals.metric=='Shares_Outstanding').filter(m.Fundamentals.ticker==ticker).one_or_none()
    shares_outstanding = utils.safe_strings_to_date(shares_outstanding.time_series,wierd=True)
    utils.dates_to_string(shares_outstanding)
    shares_outstanding = pd.Series(shares_outstanding)
    if not shares_outstanding.empty:
        close = safe_pdr_get(ticker, start=shares_outstanding.index[0], end=shares_outstanding.index[-1])
        market_cap = close * shares_outstanding
        #ev_to_ebitda
        ebitda = pd.Series(session.query(m.Fundamentals).filter(m.Fundamentals.metric=='EBITDA').filter(m.Fundamentals.ticker==ticker).one_or_none().time_series).dropna()
        shortTermDebt = pd.Series(session.query(m.Fundamentals).filter(m.Fundamentals.metric=='Short term debt').filter(m.Fundamentals.ticker==ticker).one_or_none().time_series).dropna()
        longTermDebt = pd.Series(session.query(m.Fundamentals).filter(m.Fundamentals.metric=='Long Term Debt').filter(m.Fundamentals.ticker==ticker).one_or_none().time_series).dropna()
        c_and_c_equivs = pd.Series(session.query(m.Fundamentals).filter(m.Fundamentals.metric=='Cash and Cash Equivalents').filter(m.Fundamentals.ticker==ticker).one_or_none().time_series).dropna()
        ev = shortTermDebt + longTermDebt + market_cap - c_and_c_equivs
        ebitda_to_ev = ebitda / ev
        ebitda_to_ev = ebitda_to_ev.dropna()

        #Book to price
        totalAssets = pd.Series(session.query(m.Fundamentals).filter(m.Fundamentals.metric=='Total Assets').filter(m.Fundamentals.ticker==ticker).one_or_none().time_series).dropna() 
        totalLiab = pd.Series(session.query(m.Fundamentals).filter(m.Fundamentals.metric=='Total Liabilities').filter(m.Fundamentals.ticker==ticker).one_or_none().time_series).dropna()
        book_to_price = (totalAssets - totalLiab) / shares_outstanding
        book_to_price = book_to_price.dropna()

        fcf = pd.Series(session.query(m.Fundamentals).filter(m.Fundamentals.metric=='Free Cash Flow').filter(m.Fundamentals.ticker==ticker).one_or_none().time_series).dropna()
        fcf_yield = fcf / close
        fcf_yield = fcf_yield.dropna()

        values = ebitda_to_ev * book_to_price * fcf_yield
        #print(values)
        values = values.dropna()
        if not values.empty:
            print(values[-1])
            values.index = values.index.strftime('%Y-%m-%d')
            val_to_dict = values.to_dict()
            fund = m.Fundamentals(ticker=ticker, metric='VALUE', time_series=val_to_dict)
            session.add(fund)
            print('Added ', ticker)
        else: 
            print('no value data')
            no_shares +=1     
    else:
        no_shares += 1
        print('No shares_outstanding data')
session.commit()
print('completed with {} errors'.format(no_shares))