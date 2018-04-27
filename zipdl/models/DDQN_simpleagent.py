#code adapted from https://github.com/keon/deep-q-learning/blob/master/ddqn.py
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import random

MAX_REPLAY_MEM = 2000 #Number of replays to store
GAMMA = 0.95 #Discount Rate
EPSILON = 1 #Start explore rate
MIN_EPSILON = 0.01 #Min explore rate
EPSILON_DECAY = 0.99 #Rate of exploration decay
LEARNING_RATE = 0.0005 
LAYER_DIMENSION = 32

class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MAX_REPLAY_MEM)
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.min_epsilon = MIN_EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = LEARNING_RATE
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.last_score = 0
            
    def _huber_loss(self, target, prediction):
        error = prediction-target
        return K.mean(K.sqrt(1 + K.square(error))-1, axis=-1)
    
    def _build_model(self):
        model = Sequential()
        model.add(Dense(LAYER_DIMENSION, input_dim=self.state_size, activation='relu'))
        model.add(Dense(LAYER_DIMENSION, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                        optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.reshape(state, [1, self.state_size])
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            print(state, action, reward, next_state)
            state = np.reshape(state, [1, self.state_size])
            next_state = np.reshape(next_state, [1, self.state_size])
            print(state.shape)
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state)[0]
                #print('next state', a, np.argmax(a))
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)
    
    def save(self, name):
        self.model.save_weights(name)