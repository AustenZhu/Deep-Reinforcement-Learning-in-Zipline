import numpy as numpy
import pyfolio
import zipline


from zipline.finance import commission, slippage
from zipline.pipeline import Pipeline, CustomFactor
from zipline.pipline.filters import StaticAssets
from zipline.api import set_commission, get_open_orders, symbol, sid, order_target_percent, record, get_datetime, attach_pipeline, schedule_function
from zipline.pipeline.data import USEquityPricing
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

NORMALIZE_VALUE_SCORES = False

# Upper/lower scores required for RSI signal
RSI_LOWER = 30
RSI_UPPER = 70

# Percentile in range [0, 1] of stocks that are shorted/bought
SHORTS_PERCENTILE = 0.05
LONGS_PERCENTILE = 0.05

# Constraint Parameters
MAX_GROSS_LEVERAGE = 1.0

DOLLAR_NEUTRAL = False

# If True, will screen out companies that have earnings releases between rebalance periods
AVOID_EARNINGS = True

# If False, shorts won't be ordered
ALLOW_SHORTS = True

# If True, will cut positions causing losses of LOSS_THRESHOLD or more
CUT_LOSSES = False

# Positions losing this much or more as a fraction of the investment will be cut if CUT_LOSSES is True
LOSS_THRESHOLD = 0.03


#==================PRIMARY==========================

def initialize_environment(weight, window_length, trading_start):
    def initialize(context):
        set_comission(commission.Pershare(cost=0.005, min_trade_cost=1.00))
        context.universe = utils.get_current_universe(trading_start)
        context.Factor_weights = weight
        context.window_length = window_length
        context.curr_date = trading_start
        context.mask = StaticAssets(context.universe)


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
    if not context.run_pipeline:
        return

    # Do some pre-work on factors
    if NORMALIZE_VALUE_SCORES:
        normalizeValueScores(context)
        
    if not ST_SHORTS:
        # Get indices where Sentiment Score < 0
        invalid_indices = (context.output['sentiment_score'] < 0)
        # Set those indices to 0
        context.output.loc[invalid_indices, 'sentiment_score'] = 0.0
    
    context.output = context.output.drop(['industry', 'tweets_avg'], axis=1)
    
    if PRINT_PIPE:
        log.info(context.output.head())
        log.info(len(context.output))
    
    # Rank each column of pipeline output (higher rank is better). Then create composite score based on weighted average of factor ranks
    individual_ranks = context.output.rank()
    individual_ranks *= context.Factor_weights
    ranks = individual_ranks.sum(axis=1).dropna().sort_values() + 1

    
    number_shorts = int(SHORTS_PERCENTILE*len(ranks))
    number_longs = int(LONGS_PERCENTILE*len(ranks))
    
    if number_shorts == 1 or number_longs == 1:
        number_shorts = number_longs = 0

    if (number_shorts + number_longs) > len(ranks):
        ratio = float(number_longs)/number_shorts
        number_longs = int(ratio*len(ranks)) - 1
        number_shorts = len(ranks) - number_longs
        
    shorts = 1.0 / ranks.head(number_shorts)
    shorts /= sum(shorts)
    shorts *= -1
    
    longs = ranks.tail(number_longs)
    longs /= sum(longs)
    
    if ALLOW_SHORTS:
        context.weights = shorts.append(longs)
    else:
        context.weights = longs
    # log.info(context.weights)
        
    context.output = context.output.rename(columns={'momentum_score':'momentum','sentiment_score': 'sentiment', 'value_score': 'value'})
    print("WEIGHTS: \n" + str(context.weights))
    
    comp_scores = context.output * context.Factor_weights
    comp_scores = comp_scores.sum(axis=1)
    comp_scores.name = 'composite'
    scores = pd.concat([comp_scores, context.output], axis=1)
    scores = scores.fillna(value=0.0)
    scores = scores.sort_values('composite').round(4)
    print("FACTOR SCORES: \n" + str(scores.loc[context.weights.index, :]))

    ranks.name = 'ranks'
    ranked_scores = pd.concat([ranks, comp_scores, context.output], axis=1)
    ranked_scores = ranked_scores.fillna(value=0.0)
    ranked_scores = ranked_scores.sort_values('ranks')
    ranked_scores = ranked_scores.drop(['ranks'], axis=1)
    ranked_scores = ranked_scores.round(4)
    print("BOTTOM SCORES: \n" + str(ranked_scores.head(10)))
    print("TOP SCORES: \n" + str(ranked_scores.tail(10)))
    print("Number in universe" + str(len(ranked_scores)))
    

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
    inputs = [USEquityPricing.close]
    window_length = 1
    mask = context.mask
    
    def compute(self, today, asset_ids, out, close):
        context.curr_date = today.astype(dt.datetime) 
        stocks = [symbol(ticker).sid for ticker in context.universe]
        #for verification
        print(asset_ids, stocks) 
        
        shares_outstanding = pd.Series([utils.get_fundamental(context.curr_date, 'Shares_Outstanding', ticker) for ticker in context.universe], stocks).sort_index()
        prices = pd.Series(close, asset_ids).sort_index
        market_cap = prices * shares_outstanding
        #ev_to_ebitda
        ebitda = pd.Series([utils.get_fundamental(context.curr_date, 'EBITDA', ticker) for ticker in context.universe], stocks).sort_index()
        shortTermDebt = pd.Series([utils.get_fundamental(context.curr_date, 'Short term debt', ticker) for ticker in context.universe], stocks).sort_index()
        longTermDebt = pd.Series([utils.get_fundamental(context.curr_date, 'Long Term Debt', ticker) for ticker in context.universe], stocks).sort_index()
        c_and_c_equivs = pd.Series([utils.get_fundamental(context.curr_date, 'Cash and Cash Equivalents', ticker) for ticker in context.universe], stocks).sort_index()
        ev = shortTermDebt + longTermDebt + market_cap - c_and_c_equivs
        ebitda_to_ev = ebitda / ev
        
        #Book to price
        totalAssets = pd.Series([utils.get_fundamental(context.curr_date, 'Total Assets', ticker) for ticker in context.universe], stocks).sort_index()
        totalLiab = pd.Series([utils.get_fundamental(context.curr_date, 'Total Liabilities', ticker) for ticker in context.universe], stocks).sort_index()
        book_to_price = (totalAssets - totalLiab) / shares_outstanding

        fcf = pd.Series([utils.get_fundamental(context.curr_date, 'Free Cash Flow', ticker) for ticker in context.universe], stocks).sort_index()
        fcf_yield = fcf / close

        values = ebitda_to_ev * book_to_price * fcf_yield
        ordering = [values[sid] for sid in asset_ids] 

        out[:] = ordering

        #Preparation for next trading day
        context.universe = utils.get_current_universe(context.curr_date + dt.timedelta(1))
        context.mask = StaticAssets(context.universe)

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
#============================Pipeline Stuff=============================
def make_pipeline():
    base_universe = np.array(pd.read_csv('../data/universe.csv'))
    value_factor = ValueFactor()
    momentum_factor = MomentumFactor()

    pipe = Pipeline(
        columns = {
            'value_score': value_factor,
            'momentum_score': momentum_factor,
        }
    )
    return pipe
def prime_pipeline(context, data):
    context.weeks_since_rebalance += 1
    if context.weeks_since_rebalance >= REBALANCE_PERIOD:
        context.run_pipeline = True
        context.weeks_since_rebalance = 0