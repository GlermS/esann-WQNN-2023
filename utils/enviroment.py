import random as rnd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pvlib
import sys


class EGreedyPolicy:
    def __init__(self, epsilon) -> None:
        self.epsilon = epsilon
        
    def update(self):
        pass
    def choose_action(self, Q_table):
        r = rnd.random()
        if r < self.epsilon:
            # print(state, file=sys.stderr)
            return np.argmax(Q_table)
        else:
            return np.random.choice(np.arange(0,3))

class DecayGreedyPolicy:
    def __init__(self, init, end, k) -> None:
        self.epsilon = init
        self.init = init
        self.end = end
        self.k = k
    def update(self):
        self.epsilon += self.k * (self.end - self.epsilon)

    def choose_action(self, Q_table):
        r = rnd.random()
        if r < self.epsilon:
            # print(state, file=sys.stderr)
            return np.argmax(Q_table)
        else:
            return np.random.choice(np.arange(0,3))


class Enviroment:
    def __init__(self, df, ModelGenerator, encode_func, searchMode ={'mode':'epsilon', 'params':{'epsilon':0.95}}, decay_rate = 0.1, learning_rate=0.1):
        # print(searchMode)
        if searchMode['mode'] == 'epsilon':
            self.policy = EGreedyPolicy(searchMode['params']['epsilon'])
        elif searchMode['mode'] == 'decay':
            self.policy = DecayGreedyPolicy(
                searchMode['params']['init'], 
                searchMode['params']['end'], 
                searchMode['params']['k']
                )

        self.state = (0, 0)
        self.df = df.reset_index()
        self.encode_func=encode_func
        self.T = len(self.df)
        
        self.Q = self.startQ(ModelGenerator)

        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        
    def startQ(self, ModelGenerator):
        mdls = []
        for action in range(5):
            model = ModelGenerator()
            mdls.append(model)
        return mdls
    
    def Q_predict(self, state, action):
        Qfunction = self.Q[action]
        return Qfunction.predict(self.BinInput(state))
    
    def BinInput(self, state):
        # print(state, file=sys.stderr)
        ctx = self.getContext(state = state).copy()
        ctx['angle'] = state[1]

        return self.encode_func(ctx[['ghi','zenith','azimuth','angle']].copy())
        
    def step(self, action):
        stt = self.nextState(action)
        reward = self.getResult(stt)
        done = stt[0] > self.T
        
        return stt, reward, done
        
    def getResult(self, state):
        
        ctx = self.df.iloc[state[0]]
        
        return pvlib.irradiance.get_total_irradiance(
            surface_tilt= -state[1],
            surface_azimuth=90,
            dni=ctx['dni'],
            ghi=ctx['ghi'],
            dhi=ctx['dhi'],
            solar_zenith=-ctx['zenith'],
            solar_azimuth=ctx['azimuth'], albedo=0.2
        )['poa_global']
    
    def getAction(self, state):
        Q_table = [self.Q_predict(state, action) for action in range(5)]
        return self.policy.choose_action(Q_table)
        
    def getContext(self, state = None):
        
        if state is None:
            state = self.state
        # print(len(self.df), state[0])
        return self.df.iloc[state[0]]
        
        
    def getBestAction(self, state):
        return np.argmax([self.Q_predict(state, action) for action in range(5)])
    
    def nextState(self, action):
        state = self.state
        if action == 1:
            self.state = (state[0] + 1, min([60, state[1] + 5]))
        elif action == 2:
            self.state = (state[0] + 1, max([-60, state[1]  - 5]))
        elif action == 3:
            self.state = (state[0] + 1, max([-60, state[1]  - 10]))
        elif action == 4:
            self.state = (state[0] + 1, max([-60, state[1]  - 10]))
        elif action != 0:
            raise Exception('Invalid Action')
        else:
            self.state = (state[0] + 1, state[1])
            
        return self.state    
    
    def reset(self, t = 0, s = 0):
        self.state = (t, s)