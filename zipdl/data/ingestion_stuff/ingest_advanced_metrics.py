import pandas as pd
import numpy as np  
from zipdl.data import db
from zipdl.data import models

UNIVERSE = pd.read_csv('universe.csv')
session = db.create_session()

shares_outstanding = np.array(utils.get_fundamentals(today, 'Shares_Outstanding', tickers))
market_cap = close * shares_outstanding
#ev_to_ebitda
ebitda = np.array(utils.get_fundamentals(today, 'EBITDA', tickers))
shortTermDebt = np.array(utils.get_fundamentals(today, 'Short term debt', tickers))
longTermDebt = np.array(utils.get_fundamentals(today, 'Long Term Debt', tickers))
c_and_c_equivs = np.array(utils.get_fundamentals(today, 'Cash and Cash Equivalents', tickers))
ev = shortTermDebt + longTermDebt + market_cap - c_and_c_equivs
ebitda_to_ev = ebitda / ev

#Book to price
totalAssets = np.array(utils.get_fundamentals(today, 'Total Assets', tickers))
totalLiab = np.array(utils.get_fundamentals(today, 'Total Liabilities', tickers))
book_to_price = (totalAssets - totalLiab) / shares_outstanding

fcf = np.array(utils.get_fundamentals(today, 'Free Cash Flow', tickers))
fcf_yield = fcf / close

values = ebitda_to_ev * book_to_price * fcf_yield
#print(values)
out = pd.Series(values, symbols)

session.commit()
