#Code inspired by envs from openai-gym
import math
from datetime import datetime as dt
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt

from zipdl.utils import seeding
from zipdl.utils import utils
from zipdl.utils.spaces import Discrete, DBNode

from zipline import run_algorithm
from zipdl.algos import multi_factor

START_CASH = 5000
#TODO: Allow partitioning of the time span of data to allow for cross-validation testing
TRADING_START = dt.strptime('2004-01-01', '%Y-%m-%d')
TRADING_END = dt.strptime('2016-01-01', '%Y-%m-%d')
ENV_METRICS = ['t3m', 'personal savings', 'vix']
NUM_BUCKETS = [3, 3, 3]
#Where the first element is the starting factor weighting
FACTOR_WEIGHTS = [[0.1, 0.9], [0.3, 0.7], [0.5, 0.5], [0.7, 0.3], [0.9, 0.1]]

#Save plots of reward
SAVE_FIGS=False
RENDER = False

if RENDER:
    plt.ion()

'''
MODEL OF NODES:
      
  [1,9]<->[3,7]<->[5,5]<->[7,3]<->[9,1]
    ---------------^ ^---------------
    *Each node is also able to remain stationary
    Ie. there are three moves for every node - left, stationary, and right.

Nodes action spaces parameterized as follows: [*, *, *, *, *]
Where entry corresponds to model above, and a 1 in the entry corresponds to a valid move. 
'''

class Dynamic_beta_env:
    def __init__(self, window='month', algo):
        '''
        Default rebalancing window = 'monthly'
        algo is a tuple of the following functions:
        (
            initialize_intialize: a function to create an initialize function
                Should have parameters for window length and weights
            handle_data
            rebalance_portfolio
        )
        '''
        self.starting_cash = START_CASH
        self.window = window
        if window == 'month':
            self.timedelta = 31
        self.date = TRADING_START

        self.current_node = initialize_nodes()
        self.action_space = self.current_node.action_space
        self.observation_space = Dict({metric : Discrete(bucket_num) for metric, bucket_num in zip(ENV_METRICS, NUM_BUCKETS)})

        self.seed()
        self.state = None
        self.steps_beyond_done = None

        if RENDER:
            reset_render()
            
    def initialize_nodes():
        #Initialize nodes according to mdp, and return starting nodes
        counter = 0
        for weight in FACTOR_WEIGHTS:
            DBNode(weight, counter)
            counter += 1
        return DBNode.Nodes[3] #Starting node of [5,5]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        if action == 0: 
            next_index = (self.current_node.index - 1) % len(DBNode.Nodes)
        elif action == 2: 
            next_index = (self.current_node.index + 1) % len(DBNode.Nodes)
        self.current_node = DBNode.Nodes[next_index]    
        self.state = np.array([utls.get_metric(metric, self.date) for metric in ENV_METRICS])
        if self.date + timedelta(days=self.timedelta)  > TRADING_END:
            done = True
        
        intialize = self.algo[0](self.current_node.weights, self.window)
        handle_data = self.algo[1]
        rebalance = self.algo[2]
        start = self.date
        end = self.date + dt.timedelta(self.timedelta) + 1

        perf = run_algorithm(start, end, intialize, START_CASH,
                            handle_data=handle_data)

        self.date = self.date + dt.timedelta(days=1)
        #Reward is weekly sortino
        reward = utils.calc_sortino(perf)

        if RENDER:
            self.update_viewer(reward)

        return np.array(self.state), reward, done, {}


    def reset(self):
        '''
        Reconstruct initial state
        '''
        if SAVE_FIGS:
            self.viewer.savefig('ddqn{}'.format(self.counter))
            self.counter += 1
        if RENDER:
            reset_render()
        self.date = TRADING_START
        metrics = [utils.get_metric(metric) for metric in ENV_METRICS]
        self.state = metrics
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self):
        '''
        View the windowly rewards of each trial action
        ie. Default - view the sortino of each trial
        '''
        self.viewer.show()
    
    def reset_render(self):
        self.min_x = 0
        self.max_x = (TRADING_END - TRADING_START - self.timedelta).days #ie. the max size of the training set
        self.figure, self.ax = plt.subplots()
        self.lines, = self.ax.plot([], [], 'o')
        self.ax.set_autoscaley_on(True)
        self.ax.set_xlim(self.min_x, self.max_x)
        self.counter = 0

    def update_viewer(self, new_data):
        self.viewer.set_xdata(np.append(self.viewer.get_xdata(), self.counter))
        self.viewer.set_ydata(np.append(self.viewer.get_ydata(), new_data))
        self.ax.relim()
        self.ax.autoscale_view()
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def close(self):
        if self.viewer: self.viewer.close()