"""
This is a template algorithm on Quantopian for you to adapt and fill in.


#TODO: Convert into ranking system on zipline, without st factor
"""
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline, CustomFactor
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import AverageDollarVolume
from quantopian.pipeline.filters.morningstar import Q1500US

import talib
import numpy as np
import pandas as pd
import quantopian.optimize as opt
import datetime as dt

# Imports for StockTwits factor
from quantopian.pipeline.data.psychsignal import aggregated_twitter_withretweets_stocktwits as st
from quantopian.pipeline.factors import SimpleMovingAverage

# Imports for Value factor
from quantopian.pipeline.data import morningstar


####################
### BEGIN INPUTS ###
####################

# Weeks between a rebalance
REBALANCE_PERIOD = 2

# Lookback window, in days, for averaging tweet count used in screen
ST_SCREEN_WINDOW = 7

# What top percentile of number of average tweets over the past ST_SCREEN_WINDOW days to take
TOP_TWEETS_PERCENTILE = 0.02

# Long/short SMA windows for calculating StockTwits sentiment momentum
LONG_WINDOW = 30
SHORT_WINDOW = 10

# If False, does not take StockTwits into account for short positions.
ST_SHORTS = True

# If True, will normalize value scores of each stock w.r.t. industry it is in. Turn off for speed (though results usually worse).
NORMALIZE_VALUE_SCORES = True

# How many entries of each industry required to normalize
NORMALIZATION_THRESHOLD = 3

# Lookback window, in days, for Momentum (Bollinger Bands and RSI) factor
MOMENTUM_LOOKBACK = 30

# If True, will switch from mean-reverting to trend-following momentum
TREND_FOLLOW = False

# Upper/lower SD's required for Bollinger Band signal
BBUPPER = 1.5
BBLOWER = 1.5

# Upper/lower scores required for RSI signal
RSI_LOWER = 30
RSI_UPPER = 70

# Percentile in range [0, 1] of stocks that are shorted/bought
SHORTS_PERCENTILE = 0.05
LONGS_PERCENTILE = 0.05

# If True, will vary number of shorts/longs allowed based on VIX levels
VIX_ADJUST = False

# If above, will increase short positions, decrease longs and the opposite if below
VIX_CENTER = 15.0

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

# Weights of each factor rank when calculating composite rank. Must have one entry for each factor and sum to 1.
FACTOR_WEIGHTS = [0.4, 0.4, 0.2] #[momentum, sentiment, value]
assert sum(FACTOR_WEIGHTS) == 1, "Sum of weights for factor ranks must be 1."

# Whether or not to print pipeline output stats. For backtest speed, turn off.
PRINT_PIPE = False

##################
### END INPUTS ###
##################

# Post Function for fetch_csv where vix data from Quandl is standardized
def rename_col0(df0):
    df0 = df0.rename(columns={'Close': 'price', 'Trade Date': 'Date'})
    df0 = df0.fillna(method='ffill')
    df0 = df0[['price','sid']]
    # Shifting data by one day to avoid forward-looking bias
    return df0.shift(1)
 
def initialize(context):
    """
    Called once at the start of the algorithm.
    """
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.00))
    
    # Pulling spot VIX
    fetch_csv('https://www.quandl.com/api/v3/datasets/CHRIS/CBOE_VX1.csv?api_key=CTsJVcjxStyq8o2CQ3Ts', 
        date_column='Trade Date', 
        date_format='%Y-%m-%d',
        symbol='v',
        post_func=rename_col0)
    
    
    
    # Rebalance every Wednesay, 1 hour after market open.
    schedule_function(my_rebalance, date_rules.week_start(days_offset=2), time_rules.market_open(hours=1))
    
    # Cancel unfilled orders at start of each day
    schedule_function(cancel_open_orders, date_rules.every_day(), time_rules.market_open())
     
    # Record tracking variables at the end of each day.
    schedule_function(my_record_vars, date_rules.every_day(), time_rules.market_close())
    
    # Schedule function that will set context.run_pipeline to True right before we actually want to run pipeline. Otherwise, keep at false so pipeline is not run everyday
    schedule_function(prime_pipeline, date_rules.week_start(days_offset=1), time_rules.market_close())
    
    context.weights = None
    context.run_pipeline = False
    
    context.weeks_since_rebalance = REBALANCE_PERIOD
    
     
    # Create our dynamic stock selector.
    attach_pipeline(make_pipeline(), 'my_pipeline')
  
def rank(context)

class ValueFactor(CustomFactor):
    """
    For every stock, computes a value score for it, defined as the product of 
    its book-price, FCF-price, and EBITDA-EV ratio, where a higher value is 
    desirable.
    """
    
    inputs = [morningstar.valuation_ratios.ev_to_ebitda,
              morningstar.valuation_ratios.pb_ratio,
              morningstar.valuation_ratios.fcf_yield]
    window_length = 1
    
    def compute(self, today, asset_ids, out, ev_to_ebitda, pb_ratio, fcf_yield):
        ebitda_to_ev = 1 / ev_to_ebitda
        book_to_price = 1 / pb_ratio
        ### Do we want to multiply, average, add? Multipy for now
        out[:] = ebitda_to_ev * book_to_price * fcf_yield
        
        
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
            
            # out[i] = 0
            # if rsi[-1] < RSI_LOWER:
            #     out[i] += .5
            # elif rsi[-1] > RSI_UPPER:
            #     out[i] -= .5
            # if prices[-1] < lowerBand[-1]:
            #     out[i] += .5
            # elif prices[-1] > upperBand[-1]:
            #     out[i] -= .5
            
            out[i] = 0
            if rsi[-1] < RSI_LOWER:
                out[i] += RSI_LOWER / rsi[-1]
            elif rsi[-1] > RSI_UPPER:
                out[i] -= rsi[-1] / RSI_UPPER
                
            if prices[-1] < lowerBand[-1]:
                out[i] += lowerBand[-1] / prices[-1]
            elif prices[-1] > upperBand[-1]:
                out[i] -= prices[-1] / upperBand[-1]
                
            if TREND_FOLLOW:
                out[i] *= -1

  

def make_pipeline():
    """
    A function to create our dynamic stock selector (pipeline). Documentation on
    pipeline can be found here: https://www.quantopian.com/help#pipeline-title
    """
    
    # Base universe set to the Q1500US
    base_universe = Q1500US()
    
    final_screen = base_universe
    if AVOID_EARNINGS:
        # Creates screen to make sure companies that release earnings within the rebalance period are not included
        
        # Gets days since last *quarterly* report (Morningstar ignores annual reports)
        days_since_earnings = DaysSinceEarnings()

        # Since Morningstar ignores annual reports, if days_since_earnings > 120 we can assume an annual report was released since the last quarterly report. Also, avoid if earnings were less than a week ago
        earnings_screen = ((days_since_earnings < (90 - 7 * REBALANCE_PERIOD)) & (days_since_earnings > 7)) | ((days_since_earnings > 130) & (days_since_earnings < (180 - 7 * REBALANCE_PERIOD)))

        final_screen = final_screen & earnings_screen
    
    total_tweets_avg = calculateTweetsSMA()
    
    value_factor = ValueFactor()
    sentiment_factor = calculateSTFactor()
    momentum_factor = MomentumFactor()
    
    industry_code = morningstar.asset_classification.morningstar_industry_group_code.latest
     
    pipe = Pipeline(
        screen = earnings_screen,
        columns = {
            'value_score': value_factor,
            'sentiment_score': sentiment_factor,
            'momentum_score': momentum_factor,
            'industry': industry_code,
            'tweets_avg': total_tweets_avg,
        }
    )
    
    
    assert len(pipe.columns) - 2 == len(FACTOR_WEIGHTS), "Must assign weighting to each factor." # Subtract 2 because 'industry' and 'tweets_avg' are not used for calculating rank => implicitly given weight of 0
    
    return pipe
 
    
def prime_pipeline(context, data):
    """
    Sets context.run_pipeline to True so that on the next day, the pipeline 
    will run, and all other days it will not. Done to speed up backtests.
    """
    context.weeks_since_rebalance += 1
    if context.weeks_since_rebalance >= REBALANCE_PERIOD:
        context.run_pipeline = True
        context.weeks_since_rebalance = 0
        
    
def before_trading_start(context, data):
    """
    Called every day before market open.
    """

    if not context.run_pipeline:
        return
    
    context.output = pipeline_output('my_pipeline')
    
    # Get the top percentile of tweets required
    min_tweets = context.output['tweets_avg'].dropna().quantile(q= 1 - TOP_TWEETS_PERCENTILE)
    context.output = context.output[context.output['tweets_avg'] >= min_tweets]
    
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
    individual_ranks *= FACTOR_WEIGHTS
    ranks = individual_ranks.sum(axis=1).dropna().sort_values() + 1

    
    number_shorts = int(SHORTS_PERCENTILE*len(ranks))
    number_longs = int(LONGS_PERCENTILE*len(ranks))
    
    if number_shorts == 1 or number_longs == 1:
        number_shorts = number_longs = 0
    
    if VIX_ADJUST:
        vix = data.current('v', 'price')
        scale = vix/VIX_CENTER
        number_longs =  int(number_longs ** (1/scale))
        number_shorts = int(number_shorts ** scale)
        
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
    
    comp_scores = context.output * FACTOR_WEIGHTS
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

def normalizeValueScores(context):
    industries = np.unique(context.output['industry'])
    
    total_mean = np.mean(context.output['value_score'])
    total_sd = np.std(context.output['value_score'])
    
    # Calculate mean, sd for each industry and normalize each value factor entry accordingly based on what industry it's in
    for i in industries:
        # Returns Series of True/False values depending on whether or not the value of 'industry' at that index is i
        indices = context.output['industry'] == i
        
        # Calculate mean, sd of value_scores for those entries
        industry_subseries = context.output.loc[indices, 'value_score']
        mean = np.mean(industry_subseries)
        sd = np.std(industry_subseries)
        
        # In case not enough data to normalize 
        if len(industry_subseries) < NORMALIZATION_THRESHOLD:
            mean = total_mean
            sd = total_sd
                
        # Edits value factor at those indices
        context.output.loc[indices, 'value_score'] -= mean
        context.output.loc[indices, 'value_score'] /= sd

     
def cancel_open_orders(context, data):
    """
    Cancel orders that weren't filled from before.
    """
    for stock in get_open_orders():
        for order in get_open_orders(stock):
            cancel_order(order)
 
def my_rebalance(context,data):
    """
    Execute orders according to our schedule_function() timing. 
    """
    if not context.run_pipeline:
        return
    
    close_old_positions(context, data)
    # for stock in context.weights.index:
    #     if get_open_orders(stock):
    #         continue
            
    #     weight = context.weights[stock]
    #     if weight < 0 and not ALLOW_SHORTS:
    #         continue
    #     if data.can_trade(stock):
    #         order_target_percent(stock, context.weights[stock])
    
    # Sets our objective to maximize alpha based on the weights we receive from our factor.
    objective = opt.TargetWeights(context.weights)

    # Constraints
    # -----------
    # Constrain our gross leverage to 1.0 or less. This means that the absolute
    # value of our long and short positions should not exceed the value of our
    # portfolio.
    constrain_gross_leverage = opt.MaxGrossExposure(MAX_GROSS_LEVERAGE)

    # Constrain ourselves to allocate the same amount of capital to 
    # long and short positions.
    dollar_neutral = opt.DollarNeutral(tolerance=0.05)
    
    
    constraints = [constrain_gross_leverage]
    
    if DOLLAR_NEUTRAL:
        constraints.append(dollar_neutral)
        
        
    order_optimal_portfolio(
        objective=objective,
        constraints=constraints
    )
    
    context.run_pipeline = False
            
def close_old_positions(context, data):
    to_be_closed = pd.Series()
    for stock in context.portfolio.positions:
        if stock not in context.weights:
            to_be_closed.set_value(stock, 0.0)
            
    context.weights = to_be_closed.append(context.weights)
 
def my_record_vars(context, data):
    """
    Plot variables at the end of each day.
    """
    total_positions = len(context.portfolio.positions)
    long_count = 0
    winners = 0

    to_close = []
    
    for position in context.portfolio.positions.itervalues():
        price_diff = position.last_sale_price / position.cost_basis
        if position.amount > 0:
            long_count += 1
            if price_diff > 1:
                winners += 1
                
            elif CUT_LOSSES and price_diff < (1 - LOSS_THRESHOLD):
                to_close.append(position)
                
        elif price_diff < 1:
            winners += 1
        
        elif CUT_LOSSES and (1 / price_diff) < (1 - LOSS_THRESHOLD):
            to_close.append(position)
            
     
    for position in to_close:
        order_target_percent(position.asset, 0)
            
            
    
    short_count = total_positions - long_count
    losers = total_positions - winners

    
    # Plot the counts
    record(num_long=long_count, num_short=short_count, leverage=context.account.leverage,
           win_loss=winners-losers)#, vix=data.current('v', 'price'))
 
def handle_data(context,data):
    """
    Called every minute.
    """
    pass