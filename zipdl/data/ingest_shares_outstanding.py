import pandas as pd
import numpy as np  
from pathos.multiprocessing import ProcessPool as Pool
#from zipline.data import db
#from zipline.data import models

UNIVERSE = pd.read_csv('universe.csv')
blmbg_data = pd.read_csv('shares_outstanding_data.csv',low_memory=False, header=[0,1])
for start in range(len(blmbg_data.shape)):
    focus = blmbg_data[:, start:start+1]
    ticker = blmbg_data.index
