import numpy as numpy
import pyfolio
import zipline

from zipline.finance import commission, slippage
from zipline.pipeline import CustomFactor
from zipline.api import set_commission, get_open_orders, symbol, order_target_percent, record, get_datetime
from zipline.pipeline.data import USEquityPricing
from dynamic_beta_env import 
import talib

from zipdl.utils import utils
import datetime as dt

#from ingest_fundamentals import universe, fundamentals

# Lookback window, in days, for Momentum (Bollinger Bands and RSI) factor
MOMENTUM_LOOKBACK = 30

# If true, will switch from mean-reverting to trend-following momentum
TREND_FOLLOW = True

# Upper/lower SD's required for Bollinger Band signal
BBUPPER = 1.5
BBLOWER = 1.5

# Upper/lower scores required for RSI signal
RSI_LOWER = 30
RSI_UPPER = 70

#==================PRIMARY==========================

def initialize_environment(weight, window_length, trading_start):
    def initialize(context):
        set_comission(commission.Pershare(cost=0.005, min_trade_cost=1.00))
        context.universe = utils.get_current_universe(trading_start)
        context.Factor_weights = weight
        context.window_length = window_length
        context.curr_date = trading_start
    return initialize
#schedule trading monthly
#schedule stop loss/take gain daily

def handle_data(context, data):
    pass
    
def rebalance_portfolio(context, data):
    # rebalance portfolio
    close_old_positions(context, data)
    total_weight = np.sum(context.weights.abs())
    weights = context.weights / total_weight
    for stock, weight in weights.items():
        order_target_percent(stock, weight)

def before_trading_start(context):
    context.curr_date = context.curr_date + dt.timedelta(1)

#==================UTILS==========================

def cancel_open_orders(context, data):
    for stock in get_open_orders():
        for order in get_open_orders(stock):
            cancel_order(order)

def close_old_positions(context, data):
    to_be_closed = pd.Series()
    for stock in context.portfolio.positions:
        if stock not in context.weights:
            to_be_closed.set_value(stock, 0.0)
            
    context.weights = to_be_closed.append(context.weights)

#===================FACTORS=========================
class ValueFactor(CustomFactor):
    """
    For every stock, computes a value score for it, defined as the product of 
    its book-price, FCF-price, and EBITDA-EV ratio, where a higher value is 
    desirable.
    """
    inputs = []
    window_length = 1
    
    def compute(self, today, asset_ids, out):
        context.universe = get_current_universe(context.curr_date)
        stocks = [symbol(ticker) for ticker in context.universe]
        ebit = [utils.get_fundamental(context.curr_date, 'EBIT', ticker) for ticker in context.universe]
        ebitda_to_ev = 1 / ev_to_ebitda
        book_to_price = 1 / pb_ratio
        out[:] = ebitda_to_ev * book_to_price * fcf_yield

class MomentumFactor(CustomFactor):
    """
    Uses Bollinger Bands and RSI measures to determine whether or not a stock 
    should be bought (return 1), sold (return -1), or if there is no signal 
    (return 0). For a signal, both metrics have to indicate the same signal 
    (e.g., price below lower Bollinger Band and RSI below RSI_LOWER)
    """
    window_length = MOMENTUM_LOOKBACK+10
    inputs = [USEquityPricing.close]
    
    def compute(self, today, asset_ids, out, close):
        for i in range(len(out)):
            out[i] = 0
            prices = close[:, i]
            try:
                upperBand, middleBand, lowerBand = talib.BBANDS(
                    prices,
                    timeperiod = MOMENTUM_LOOKBACK,
                    nbdevup=BBUPPER,
                    nbdevdn=BBLOWER,
                    matype=0)
                
                rsi = talib.RSI(prices, timeperiod=MOMENTUM_LOOKBACK)
                               
            except:
                out[i] = 0
                continue
            
            out[i] = 0
            if rsi[-1] < RSI_LOWER:
                out[i] += .5
            elif rsi[-1] > RSI_UPPER:
                out[i] -= .5
            if prices[-1] < lowerBand[-1]:
                out[i] += .5
            elif prices[-1] > upperBand[-1]:
                out[i] -= .5
                
            if TREND_FOLLOW:
                out[i] *= -1