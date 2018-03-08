#Code inspired by envs from openai-gym
import math
import numpy as np
from utils import seeding

START_CASH = 5000

class Dynamic_beta_env:
    def __init__(self):
        self.starting_cash = START_CASH

        self.action_space = ...
        self.observation_space = ...

        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.