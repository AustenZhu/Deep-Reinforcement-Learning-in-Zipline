from zipdl.utils import utils
import datetime as dt

from zipline import run_algorithm
utils.clean_db_time_series(['VALUE'])
import pandas as pd
from matplotlib import pyplot as plt

start_capital = 5000

TRADING_START = pd.to_datetime('2011-01-01').tz_localize('US/Eastern')
TRADING_END = pd.to_datetime('2016-01-01').tz_localize('US/Eastern')

from zipdl.algos import multi_factor_rrn_test as mf
from zipdl.models.RRN_simpleagent import RRNAgent
algo = [mf.initialize_environment, mf.handle_data, mf.before_trading_start]
before_trading_start = algo[2]
state_size = len(mf.ENV.observation_space)
action_size = mf.ENV.action_space.n
agent = RRNAgent(state_size, action_size)
done = False

agent.load('/home/austen/sp18/CIB/dynamic_beta_research/zip_dl_infra/saved_models/dynamic-beta-rrn-10.h5')
initialize = algo[0](agent, TRADING_START)
perf5 = run_algorithm(TRADING_START, TRADING_END, 
                    initialize=initialize, 
                    capital_base=start_capital, 
                    before_trading_start=before_trading_start,
                    metrics_set='faster')

agent.load('/home/austen/sp18/CIB/dynamic_beta_research/zip_dl_infra/saved_models/dynamic-beta-rrn-20.h5')
initialize = algo[0](agent, TRADING_START)
perf6 = run_algorithm(TRADING_START, TRADING_END, 
                    initialize=initialize, 
                    capital_base=start_capital, 
                    before_trading_start=before_trading_start,
                    metrics_set='faster')
agent.load('/home/austen/sp18/CIB/dynamic_beta_research/zip_dl_infra/saved_models/dynamic-beta-rrn-30.h5')
initialize = algo[0](agent, TRADING_START)
perf7 = run_algorithm(TRADING_START, TRADING_END, 
                    initialize=initialize, 
                    capital_base=start_capital, 
                    before_trading_start=before_trading_start,
                    metrics_set='faster')

from zipdl.algos import multi_factor_ddqn_test as mf
from zipdl.models.DDQN_simpleagent import DDQNAgent

algo = [mf.initialize_environment, mf.handle_data, mf.before_trading_start]
before_trading_start = algo[2]
state_size = len(mf.ENV.observation_space)
action_size = mf.ENV.action_space.n
agent = DDQNAgent(state_size, action_size)
done = False



agent.load('/home/austen/sp18/CIB/dynamic_beta_research/zip_dl_infra/saved_models/dynamic_beta-ddqn-0.h5')
initialize = algo[0](agent, TRADING_START)
perf1 = run_algorithm(TRADING_START, TRADING_END, 
                    initialize=initialize, 
                    capital_base=start_capital, 
                    before_trading_start=before_trading_start,
                    metrics_set='faster')

agent.load('/home/austen/sp18/CIB/dynamic_beta_research/zip_dl_infra/saved_models/dynamic_beta-ddqn-10.h5')
initialize = algo[0](agent, TRADING_START)
perf2 = run_algorithm(TRADING_START, TRADING_END, 
                    initialize=initialize, 
                    capital_base=start_capital, 
                    before_trading_start=before_trading_start,
                    metrics_set='faster')

agent.load('/home/austen/sp18/CIB/dynamic_beta_research/zip_dl_infra/saved_models/dynamic_beta-ddqn-20.h5')
initialize = algo[0](agent, TRADING_START)
perf3 = run_algorithm(TRADING_START, TRADING_END, 
                    initialize=initialize, 
                    capital_base=start_capital, 
                    before_trading_start=before_trading_start,
                    metrics_set='faster')

agent.load('/home/austen/sp18/CIB/dynamic_beta_research/zip_dl_infra/saved_models/dynamic_beta-ddqn-30.h5')
initialize = algo[0](agent, TRADING_START)
perf4 = run_algorithm(TRADING_START, TRADING_END, 
                    initialize=initialize, 
                    capital_base=start_capital, 
                    before_trading_start=before_trading_start,
                    metrics_set='faster')

from zipdl.algos import mult_factor_naive as mf
algo = [mf.initialize_environment, mf.handle_data, mf.before_trading_start]
before_trading_start = algo[2]
initialize = mf.initialize_environment(TRADING_START, trading_day=3)
perf8 = run_algorithm(TRADING_START, TRADING_END,
                    initialize=initialize, 
                    capital_base=start_capital, 
                    before_trading_start = mf.before_trading_start, 
                    metrics_set='faster')

plt.figure()
plt.plot(perf1['algorithm_period_return'], label='ddqn-0')
plt.plot(perf2['algorithm_period_return'], label='ddqn-10')
plt.plot(perf3['algorithm_period_return'], label='ddqn-20')
plt.plot(perf4['algorithm_period_return'], label='ddqn-30')
plt.plot(perf5['algorithm_period_return'], label='rrn-10')
plt.plot(perf6['algorithm_period_return'], label='rrn-20')
plt.plot(perf7['algorithm_period_return'], label='rrn-30')
plt.plot(perf8['algorithm_period_return'], label='naive')
plt.legend()
plt.savefig('train_data')
print('Growth')
print('ddqn-0:', (perf8['algorithm_period_return'][-1] - perf1['algorithm_period_return'][-1])/perf8['algorithm_period_return'][-1])
print('ddqn-10:', (perf8['algorithm_period_return'][-1] - perf2['algorithm_period_return'][-1])/perf8['algorithm_period_return'][-1])
print('ddqn-20:', (perf8['algorithm_period_return'][-1] - perf3['algorithm_period_return'][-1])/perf8['algorithm_period_return'][-1])
print('ddqn-30:', (perf8['algorithm_period_return'][-1] - perf4['algorithm_period_return'][-1])/perf8['algorithm_period_return'][-1])
print('rrn-10:', (perf8['algorithm_period_return'][-1] - perf5['algorithm_period_return'][-1])/perf8['algorithm_period_return'][-1])
print('rrn-20:', (perf8['algorithm_period_return'][-1] - perf6['algorithm_period_return'][-1])/perf8['algorithm_period_return'][-1])
print('rrn-30:', (perf8['algorithm_period_return'][-1] - perf7['algorithm_period_return'][-1])/perf8['algorithm_period_return'][-1])


plt.figure()
plt.plot(perf1['sortino'][20:], label='ddqn-0')
plt.plot(perf2['sortino'][20:], label='ddqn-10')
plt.plot(perf3['sortino'][20:], label='ddqn-20')
plt.plot(perf4['sortino'][20:], label='ddqn-30')
plt.plot(perf5['sortino'][20:], label='rrn-10')
plt.plot(perf6['sortino'][20:], label='rrn-20')
plt.plot(perf7['sortino'][20:], label='rrn-30')
plt.plot(perf8['sortino'][20:], label='naive')
plt.legend()
plt.savefig('train_data_sortino')
print('sortino_change')
print('ddqn-0:', (perf8['sortino'][-1] - perf1['sortino'][-1])/perf8['sortino'][-1])
print('ddqn-10:', (perf8['sortino'][-1] - perf2['sortino'][-1])/perf8['sortino'][-1])
print('ddqn-20:', (perf8['sortino'][-1] - perf3['sortino'][-1])/perf8['sortino'][-1])
print('ddqn-30:', (perf8['sortino'][-1] - perf4['sortino'][-1])/perf8['sortino'][-1])
print('rrn-10:', (perf8['sortino'][-1] - perf5['sortino'][-1])/perf8['sortino'][-1])
print('rrn-20:', (perf8['sortino'][-1] - perf6['sortino'][-1])/perf8['sortino'][-1])
print('rrn-30:', (perf8['sortino'][-1] - perf7['sortino'][-1])/perf8['sortino'][-1])


#Testing Set:
TRADING_START = pd.to_datetime('2016-01-01').tz_localize('US/Eastern')
TRADING_END = pd.to_datetime('2018-01-01').tz_localize('US/Eastern')

from zipdl.algos import multi_factor_ddqn_test as mf
algo = [mf.initialize_environment, mf.handle_data, mf.before_trading_start]
before_trading_start = algo[2]
state_size = len(mf.ENV.observation_space)
action_size = mf.ENV.action_space.n
agent = DDQNAgent(state_size, action_size)
done = False


agent.load('/home/austen/sp18/CIB/dynamic_beta_research/zip_dl_infra/saved_models/dynamic_beta-ddqn-0.h5')
initialize = algo[0](agent, TRADING_START)
test1 = run_algorithm(TRADING_START, TRADING_END, 
                    initialize=initialize, 
                    capital_base=start_capital, 
                    before_trading_start=before_trading_start,
                    metrics_set='faster')

agent.load('/home/austen/sp18/CIB/dynamic_beta_research/zip_dl_infra/saved_models/dynamic_beta-ddqn-10.h5')
initialize = algo[0](agent, TRADING_START)
test2 = run_algorithm(TRADING_START, TRADING_END, 
                    initialize=initialize, 
                    capital_base=start_capital, 
                    before_trading_start=before_trading_start,
                    metrics_set='faster')

agent.load('/home/austen/sp18/CIB/dynamic_beta_research/zip_dl_infra/saved_models/dynamic_beta-ddqn-20.h5')
initialize = algo[0](agent, TRADING_START)
test3 = run_algorithm(TRADING_START, TRADING_END, 
                    initialize=initialize, 
                    capital_base=start_capital, 
                    before_trading_start=before_trading_start,
                    metrics_set='faster')

agent.load('/home/austen/sp18/CIB/dynamic_beta_research/zip_dl_infra/saved_models/dynamic_beta-ddqn-30.h5')
initialize = algo[0](agent, TRADING_START)
test4 = run_algorithm(TRADING_START, TRADING_END, 
                    initialize=initialize, 
                    capital_base=start_capital, 
                    before_trading_start=before_trading_start,
                    metrics_set='faster')


from zipdl.algos import multi_factor_rrn_test as mf
from zipdl.models.RRN_simpleagent import RRNAgent
algo = [mf.initialize_environment, mf.handle_data, mf.before_trading_start]
before_trading_start = algo[2]
state_size = len(mf.ENV.observation_space)
action_size = mf.ENV.action_space.n
agent = RRNAgent(state_size, action_size)
done = False

agent.load('/home/austen/sp18/CIB/dynamic_beta_research/zip_dl_infra/saved_models/dynamic-beta-rrn-10.h5')
initialize = algo[0](agent, TRADING_START)
test5 = run_algorithm(TRADING_START, TRADING_END, 
                    initialize=initialize, 
                    capital_base=start_capital, 
                    before_trading_start=before_trading_start,
                    metrics_set='faster')
agent.load('/home/austen/sp18/CIB/dynamic_beta_research/zip_dl_infra/saved_models/dynamic-beta-rrn-20.h5')
initialize = algo[0](agent, TRADING_START)
test6 = run_algorithm(TRADING_START, TRADING_END, 
                    initialize=initialize, 
                    capital_base=start_capital, 
                    before_trading_start=before_trading_start,
                    metrics_set='faster')

agent.load('/home/austen/sp18/CIB/dynamic_beta_research/zip_dl_infra/saved_models/dynamic-beta-rrn-30.h5')
initialize = algo[0](agent, TRADING_START)
test7 = run_algorithm(TRADING_START, TRADING_END, 
                    initialize=initialize, 
                    capital_base=start_capital, 
                    before_trading_start=before_trading_start,
                    metrics_set='faster')

from zipdl.algos import mult_factor_naive as mf
algo = [mf.initialize_environment, mf.handle_data, mf.before_trading_start]
before_trading_start = algo[2]
initialize = mf.initialize_environment(TRADING_START, trading_day=3)
test8 = run_algorithm(TRADING_START, TRADING_END,
                    initialize=initialize, 
                    capital_base=start_capital, 
                    before_trading_start = mf.before_trading_start, 
                    metrics_set='faster')




plt.figure()
plt.plot(test1['algorithm_period_return'], label='ddqn-0')
plt.plot(test2['algorithm_period_return'], label='ddqn-10')
plt.plot(test3['algorithm_period_return'], label='ddqn-20')
plt.plot(test4['algorithm_period_return'], label='ddqn-30')
plt.plot(test5['algorithm_period_return'], label='rrn-10')
plt.plot(test6['algorithm_period_return'], label='rrn-20')
plt.plot(test7['algorithm_period_return'], label='rrn-30')
plt.plot(test8['algorithm_period_return'], label='naive')
plt.legend()
plt.savefig('test_data')
print('Growth')
print('ddqn-0:', (test8['algorithm_period_return'][-1] - test1['algorithm_period_return'][-1])/test8['algorithm_period_return'][-1])
print('ddqn-10:', (test8['algorithm_period_return'][-1] - test2['algorithm_period_return'][-1])/test8['algorithm_period_return'][-1])
print('ddqn-20:', (test8['algorithm_period_return'][-1] - test3['algorithm_period_return'][-1])/test8['algorithm_period_return'][-1])
print('ddqn-30:', (test8['algorithm_period_return'][-1] - test4['algorithm_period_return'][-1])/test8['algorithm_period_return'][-1])
print('rrn-10:', (test8['algorithm_period_return'][-1] - test5['algorithm_period_return'][-1])/test8['algorithm_period_return'][-1])
print('rrn-20:', (test8['algorithm_period_return'][-1] - test6['algorithm_period_return'][-1])/test8['algorithm_period_return'][-1])
print('rrn-30:', (test8['algorithm_period_return'][-1] - test7['algorithm_period_return'][-1])/test7['algorithm_period_return'][-1])

plt.figure()
plt.plot(test1['sortino'], label='ddqn-0')
plt.plot(test2['sortino'], label='ddqn-10')
plt.plot(test3['sortino'], label='ddqn-20')
plt.plot(test4['sortino'], label='ddqn-30')
plt.plot(test5['sortino'], label='rrn-10')
plt.plot(test6['sortino'], label='rrn-20')
plt.plot(test7['sortino'], label='rrn-30')
plt.plot(test8['sortino'], label='naive')
plt.legend()
plt.savefig('test_data_sortino')
print('ddqn-0:', (test8['sortino'][-1] - test1['sortino'][-1])/test8['sortino'][-1])
print('ddqn-10:', (test8['sortino'][-1] - test2['sortino'][-1])/test8['sortino'][-1])
print('ddqn-20:', (test8['sortino'][-1] - test3['sortino'][-1])/test8['sortino'][-1])
print('ddqn-30:', (test8['sortino'][-1] - test4['sortino'][-1])/test8['sortino'][-1])
print('rrn-10:', (test8['sortino'][-1] - test5['sortino'][-1])/test8['sortino'][-1])
print('rrn-20:', (test8['sortino'][-1] - test6['sortino'][-1])/test8['sortino'][-1])
print('rrn-30:', (test8['sortino'][-1] - test7['sortino'][-1])/test8['sortino'][-1])


