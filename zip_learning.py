from zipdl.algos import multi_factor_training as mf
from zipdl.utils import utils
import datetime as dt
from zipdl.models.DDQN_agent import DDQNAgent

from zipline import run_algorithm
utils.clean_db_time_series(mf.DB_FACTORS_USED)
import pandas as pd

BATCH_SIZE = 32
EPISODES = 5000
start_capital = 5000

TRADING_START = pd.to_datetime('2010-01-01').tz_localize('US/Eastern')
TRADING_END = pd.to_datetime('2016-01-01').tz_localize('US/Eastern')

algo = [mf.initialize_environment, mf.handle_data, mf.before_trading_start]
state_size = len(mf.ENV.observation_space)
action_size = mf.ENV.action_space.n
agent = DDQNAgent(state_size, action_size)
done = False

'''
1 episode test train

initialize = algo[0](agent)
perf = run_algorithm(TRADING_START, TRADING_END, 
                    initialize=initialize, 
                    capital_base=start_capital, 
                    before_trading_start=before_trading_start,
                    metrics_set='faster')
#NOTE - all training is done within the algo. 
'''

for e in range(EPISODES):
    state = mf.ENV.reset()
    run_algorithm(TRADING_START, TRADING_END, 
                    initialize=initialize, 
                    capital_base=start_capital, 
                    before_trading_start=before_trading_start,
                    metrics_set='faster')
    agent.update_target_model()
    print("episode: {}/{}, score: {}, e: {:.2}"
            .format(e, EPISODES, time, agent.epsilon))
    # if e % 10 == 0:
    #     agent.save("./save/cartpole-ddqn.h5")
