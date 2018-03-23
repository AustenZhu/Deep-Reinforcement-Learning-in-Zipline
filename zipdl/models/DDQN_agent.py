#code adapted from https://github.com/keon/deep-q-learning/blob/master/ddqn.py
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import numpy as np

from zipline import run_algorithm

MAX_REPLAY_MEM = 2000 #Number of replays to store
GAMMA = 0.95 #Discount Rate
EPSILON = 1 #Start explore rate
EPMIN_EPSILON = 0.01 #Min explore rate
EPSILON_DECAY = 0.99 #Rate of exploration decay
LEARNING_RATE = 0.001 
LAYER_DIMENSION = 24

SORTINO_GOAL = 1.2

class DDQNAgent:
    def __init__(self, state_size, action_size, algo):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MAX_REPLAY_MEM)
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.min_epsilon = MIN_EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = LEARNING_RATE
        self.model = self._build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        self.algo = algo

    def _finance_loss(self, model_metric, goal=SORTINO_GOAL):
        '''
        Huber loss with target as the metric 
        (We look for a constant adjusted rate of risk-reward,
            and reward any performance higher than that)
        '''
        #TODO: Calculate metric
        return self._huber_loss(goal, model)
            
    def _huber_loss(self, target, prediction):
        error = prediction-target
        return K.mean(K.sqrt(1 + K.square(error))-1, axis=-1)
    
    def _build_model(self):
        #TODO - Calculate largest action space for dbnodes, then dropout neurons that will not be used for other nodes
        model = Sequential()
        model.add(Dense(LAYER_DIMENSION, input=self.state_size, activation='relu'))
        model.add(Dense(LAYER_DIMENSION, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._finance_loss,
                        optimizer=Adam(lr=self.learning_rate))
        return models

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)
    
    def save(self, name):
        self.model.save_weights(name)