'''
BEFORE RUNNING ANYTHING!
#Make sure to download simfin fundamentals data first!

Code for ingesting fundamental data for the custom universe
ie. the universe of the 1500 companies that we have data on

Metrics included (get by accessing the dictionary fundamentals:)
Gross Margin
Cash From Investing Activities
Minorities
Free Cash Flow
Total Noncurrent Assets
Net PP&E
Income Taxes
Depreciation & Amortisation
Change in Working Capital
Equity Before Minorities
Cash and Cash Equivalents
SG&A
Goodwill
Cash From Operating Activities
Return on Equity
Retained Earnings
Current Assets
Net Change in PP&E & Intangibles
Dividends
Accounts Payable
Return on Assets
Net Change in Cash
Current Ratio
Net Income from Discontinued Op.
Treasury Stock
Total Noncurrent Liabilities
Share Capital
Revenues
Long Term Debt
EBITDA
Total Assets
Total Liabilities
Intangible Assets
EBIT
Net Profit Margin
R&D
Interest expense, net
COGS
Liabilities to Equity Ratio
Abnormal Gains/Losses
Net Profit
Preferred Equity
Total Equity
Debt to Assets Ratio
Current Liabilities
Receivables
Operating Margin
Short term debt
Cash From Financing Activities
'''

import pandas as pd
import numpy as np  
from pathos.multiprocessing import ProcessPool as Pool
from zipdl.data import db
from zipdl.data import models

NPROC = 8

fundamentals = {}
def process_fundamental(metric):
    '''
    df - input dataframe, 
    metric - what metric to process
    fundamentals - what dictionary to join to
    join_universe - what list to join to
    '''
    df = reduce_df(full_df, metric)
    df = df.set_index(list(df)[0])
    print('Starting ingestion of {}'.format(metric))
    
    store = {}
    for ticker in set(df):
        if not all(df[ticker].isnull()):
            series = df[ticker].dropna()
            store[ticker] = series
    print('Ingested {}'.format(metric))
    #print(store)
    return store

def reduce_df(df, metric):
    temp = df[df.columns.drop([label for label in df.columns if label[1] != metric and label[1] != date_col[1]])]
    #print(temp.columns) 
    temp.columns = temp.columns.droplevel(level=1)
    return temp

print('Beginning... ')
full_df = pd.read_csv('../../../../fundamentals.csv', low_memory=False, header=[0,1])
print('csv in memory')

metrics = set([x[1] for x in list(full_df)[1:]])
date_col = list(full_df)[0]

pool = Pool(nodes=NPROC)
func_args = list(metrics)
print('finished creating args')
try:
    outputs = pool.uimap(process_fundamental, func_args)
except Exception as e:
    print(e)


for metric, data in zip(func_args, outputs):
    fundamentals[metric] = data

#TODO: Export data into db for faster access
#models.Base.metadata.create_all(db.engine)
session = db.create_session()
with open('universe.csv') as f:
    data = f.read().replace("'", '').replace(' ','').replace('\n', '').split(',')
universe = data
for fundamental in func_args:
    current = fundamentals[fundamental]
    for ticker in universe:
        real = ticker.replace("'", '').replace(' ','').replace('\n', '')
        data = current[real].to_dict()
        fnd = models.Fundamentals(ticker=ticker, metric=fundamental, time_series=data)
        session.add(fnd)
    session.commit()
    