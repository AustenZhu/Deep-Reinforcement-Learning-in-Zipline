from zipdl.envs import dynamic_beta_env as dbenv

ENV = dbenv.Dynamic_beta_env(dbenv.TRADING_START)
'''
ALGO BELOW
'''
import numpy as np
import pandas as pd
from numexpr import evaluate
import zipline

from zipline.pipeline.factors.technical import RSI
from zipline.finance import commission, slippage
from zipline.pipeline import Pipeline, CustomFactor
from zipline.pipeline.filters import StaticAssets
from zipline.data import bundles
from zipline.api import *
from zipline.pipeline.data import USEquityPricing
from zipline.utils.math_utils import nanmean, nanstd
from zipline.finance.slippage import FixedSlippage

import empyrical
from zipdl.utils import utils
import datetime as dt
from sklearn.preprocessing import minmax_scale

# Weeks between a rebalance
REBALANCE_PERIOD = 4

# Lookback window, in days, for Momentum (Bollinger Bands and RSI) factor
MOMENTUM_LOOKBACK = 30

# If true, will switch from mean-reverting to trend-following momentum
TREND_FOLLOW = True

# Upper/lower SD's required for Bollinger Band signal
BBUPPER = 1.5
BBLOWER = 1.5
BBSTD = 1.5

NORMALIZE_VALUE_SCORES = False

# Upper/lower scores required for RSI signal
RSI_LOWER = 30
RSI_UPPER = 70

# Percentile in range [0, 1] of stocks that are shorted/bought
SHORTS_PERCENTILE = 0.01
LONGS_PERCENTILE = 0.01

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

# Whether or not to print pipeline output stats. For backtest speed, turn off.
PRINT_PIPE = False

BATCH_SIZE = 32

ACTION = 0
#=================util=============================
#Throws out tickers not found in quandl-quantopian data (mostly tickers with no vol)
def safe_symbol(ticker):
    try:
        x = symbol(ticker)
        return x
    except: 
        return None
def safe_symbol_convert(tickers):
    filtered_list = list(filter(None.__ne__, [safe_symbol(ticker) for ticker in tickers]))
    return filtered_list

def universe_transform(date):
    universe = utils.get_current_universe(date)
    tradable_assets = safe_symbol_convert(universe)
    return tradable_assets

#==================PRIMARY==========================
def initialize_environment(agent, trading_day=2):
    assert 1 <= trading_day <= 5
    def initialize(context):
        context.agent = agent
        context.values = deque(maxlen=21)
        set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.00))
        set_slippage(slippage.FixedSlippage(0.00))
        context.universe = universe_transform('2018-01-01')

        schedule_function(rebalance_portfolio, date_rules.week_start(days_offset=trading_day), time_rules.market_open(hours=1))
        
        #schedule_function(cancel_open_orders, date_rules.every_day(), time_rules.market_open())

        schedule_function(prime_pipeline, date_rules.week_start(days_offset=trading_day-1), time_rules.market_close())

        context.Factor_weights = ENV.current_node.weights
        context.weights = None
        context.run_pipeline = True #We want to run stock selector immediately
        context.weeks_since_rebalance = -1

        attach_pipeline(make_pipeline(context), 'my_pipeline')


    return initialize
#schedule trading monthly
#schedule stop loss/take gain daily

def handle_data(context, data):
    #Daily function
    #context.universe = universe_transform(get_datetime())
    context.values.append(context.portfolio.portfolio_value)

def rebalance_portfolio(context, data):
    # rebalance portfolio
    close_old_positions(context, data)
    total_weight = np.sum(context.weights.abs())
    weights = context.weights / total_weight
    for stock, weight in weights.items():
        result = order_target_percent(stock, weight)
        
def before_trading_start(context, data):
    if not context.run_pipeline:
        return

    date = get_datetime()
    if (date - context.start_date).days > 12:
        print('training on {}'.format(date))
        returns = pd.Series(list(context.values)).pct_change()
        context.values.clear()
        sortino_reward = empyrical.sortino_ratio(returns, period=empyrical.MONTHLY)
        ENV.update_state(date)
        context.agent.remember(ENV.prev_state, ACTION, sortino_reward, ENV.state, False)
        ACTION = context.agent.act(ENV.state)
        context.Factor_weights = ENV.step(ACTION)
        if len(context.agent.memory) > BATCH_SIZE:
            context.agent.replay(BATCH_SIZE)
        
    context.run_pipeline = False
    def compute(today, symbols):
        #for verification
        tickers = [symbol.symbol for symbol in symbols]
        values = np.array(utils.get_fundamentals(today, 'VALUE', tickers))
        #print(values)
        values = minmax_scale(values, feature_range=(-1,1))
        out = pd.Series(values, symbols)
        return out
    value_factor = compute(context.curr_date, context.universe).to_frame()

    context.output = pipeline_output('my_pipeline')
    context.output = context.output.join(value_factor).dropna()
    
    # Do some pre-work on factors
    if NORMALIZE_VALUE_SCORES:
        normalizeValueScores(context)
    
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
DB_FACTORS_USED = ['Shares_Outstanding', 'EBITDA', 'Short term debt', 'Long Term Debt', 'Cash and Cash Equivalents',
                    'Total Assets', 'Total Liabilities', 'Free Cash Flow']

class MomentumFactor(CustomFactor):
    """
    Uses Bollinger Bands and RSI measures to determine whether or not a stock 
    should be bought (return 1), sold (return -1), or if there is no signal 
    (return 0). For a signal, both metrics have to indicate the same signal 
    (e.g., price below lower Bollinger Band and RSI below RSI_LOWER)
    """
    inputs = [USEquityPricing.close]
    window_length = MOMENTUM_LOOKBACK+10
    
    def compute(self, today, asset_ids, out, close):
        diffs = np.diff(close, axis=0)
        ups = nanmean(np.clip(diffs, 0, np.inf), axis=0)
        downs = abs(nanmean(np.clip(diffs, -np.inf, 0), axis=0))
        rsi = np.zeros(len(out))
        evaluate(
            "100 - (100 / (1 + (ups / downs)))",
            local_dict={'ups': ups, 'downs': downs},
            global_dict={},
            out=rsi,
        )
        
        difference = BBSTD * nanstd(close, axis=0)
        middle = nanmean(close, axis=0)
        upper = middle + difference
        lower = middle - difference

        for i in range(len(out)):
            out[i] = 0
            if rsi[i] < RSI_LOWER:
                out[i] += RSI_LOWER / rsi[i]
            elif rsi[i] > RSI_UPPER:
                out[i] -= rsi[i] / RSI_UPPER
                
            prices = close[:, i]
            if prices[-1] < lower[i]:
                out[i] += lower[i] / prices[-1]
            elif prices[-1] > upper[i]:
                out[i] -= prices[-1] / upper[i]
                
            if TREND_FOLLOW:
                out[i] *= -1
#============================Pipeline Stuff=============================
def make_pipeline(context):
    
    screen = StaticAssets(context.universe)

    momentum_factor = MomentumFactor()
    pipe = Pipeline(
        columns = {
            'momentum_score': momentum_factor,
        }
    )
    pipe.set_screen(screen)
    return pipe
def prime_pipeline(context, data):
    context.weeks_since_rebalance += 1
    if context.weeks_since_rebalance >= REBALANCE_PERIOD:
        context.run_pipeline = True
        context.weeks_since_rebalance = 0

