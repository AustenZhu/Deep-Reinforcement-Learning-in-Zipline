import pandas as pd
import numpy as np  
import db
import models

UNIVERSE = pd.read_csv('universe.csv')
blmbg_data = pd.read_csv('shares_outstanding_data.csv',low_memory=False, header=[0,1])
session = db.create_session()
for start in range(int(blmbg_data.shape[1]/2)):
    print('Starting on {}'.format(start))
    index = start * 2
    focus = blmbg_data.iloc[:, index:index+2]
    focus = focus.set_index(list(focus)[0])
    focus.columns = focus.columns.droplevel(level=1)
    focus.index.names = ['Date']
    ticker = focus.columns[0].replace('US EQUITY', '').replace(' ', '')
    focus_series = focus.squeeze().to_dict()
    fnd = models.Fundamentals(ticker=ticker, metric='Shares_Outstanding', time_series=focus_series)
    session.add(fnd)
    print('Added {}'.format(ticker))
session.commit()
