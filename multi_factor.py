import numpy as numpy
import pyfolio
from zipline.finance import commission, slippage
from zipline.pipeline import CustomFactor
from zipline.api import set_commission
from zipline.pipeline.data import USEquityPricing
import talib

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

def initialize_environment(weight):
    def initialize(context):
        set_comission(commission.Pershare(cost=0.005, min_trade_cost=1.00))
        context.Factor_weights = get_factor_weights()
    return initialize

def handle_data(context, data):
    order(symbol('AAPL'), 10)

class ValueFactor(CustomFactor):
    """
    For every stock, computes a value score for it, defined as the product of 
    its book-price, FCF-price, and EBITDA-EV ratio, where a higher value is 
    desirable.


    """
    inputs = [fundamentals['EBITDA'],
              fundamentals['Gross Margin'],
              fundamentals['Free Cash Flow']]
    window_length = 1
    
    def compute(self, today, asset_ids, out, ev_to_ebitda, pb_ratio, fcf_yield):
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