'''
TODO: Grab metric from db

'''
import numpy as np


def get_metric(metric):
    pass

def calc_annualized_weekly_sortino(perf, risk_free_rate=0.00136):
    returns = perf['portfolio_value'][1::5].pct_change().dropna()


def calc_sortino(perf):
    returns = perf['portfolio_value'][::5].pct_change()[1:]
    weekly_mar = 0.1 ** (1/52) #10% annualized minimum accepted return
    #Target downside deviation, as calculated here: https://www.sunrisecapital.com/wp-content/uploads/2013/02/Futures_Mag_Sortino_0213.pdf
    tdd = np.sqrt(np.sum([min(0, returns - weekly_mar)]))
    return (returns.mean() - weekly_mar)/tdd