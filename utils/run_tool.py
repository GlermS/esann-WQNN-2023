from utils.enviroment import Enviroment
import utils.solar as sol
from pywisard.binarization.thermometer import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import utils.plotUtils as pu


def init(dfCs):
    
    dfCs['angle'] = np.nan
    dfCs['month'] = dfCs.index.month
    dfCs['day'] = dfCs.index.day
    dfCs['hour'] = dfCs.index.hour
    dfCs['minute'] = dfCs.index.minute/12
    dfCs['time'] = dfCs.index.hour*60 + dfCs.index.minute

    encoders = {
        'ghi' : DistributiveEncoder(bins=600).fit(dfCs.query('mghi > 0')['mghi']),
        'zenith' : DistributiveEncoder(bins=120*2).fit(dfCs['zenith']),
        'azimuth' : DistributiveEncoder(bins=120*2).fit(dfCs['azimuth']),
        'angle' : LinearEncoder(bins=120*2).fit(dfCs['mangle']),
        'month' : CircularEncoder(bins=len(dfCs['month'].unique())).fit(dfCs['month']),
        'time' : LinearEncoder(bins=288).fit(dfCs['mangle'])
    }

    return dfCs, encoders


def run_individual(dfCs, ModelGenerator, encode_func, path, decay_rate = 0.6, learning_rate=1, searchMode = {'mode':'epsilon', 'params':{'epsilon':0.95}}, index=0):
    variables_to_store =['ts','mg','POA', 'mghi','ghi','mangle','angle']
    
    env = Enviroment(dfCs, ModelGenerator, encode_func, decay_rate = decay_rate, learning_rate=learning_rate, searchMode = searchMode)
    done = False
    env.reset()
    state = env.state
    
    dfr = dfCs.copy().reset_index()
    print(dfr.columns)
    try:
        while done == False:
            if state[0] >= len(dfCs):
                break
            action = env.getAction(state)
            nextState, reward, done = env.step(action)

            ctx = env.getContext()
            bonus = 0 if ctx.hour < 3 and np.abs(ctx.angle) < 10 else -500
    
            Q  = env.Q_predict(state, action)
            nextAction = env.getBestAction(nextState)
            Q_ = env.Q_predict(nextState, nextAction)
            
            eps = env.learning_rate
            gamma = env.decay_rate
            env.Q[action].train(env.BinInput(state), (1 - eps) * Q + eps * ((reward+bonus)**3  + gamma * Q_))
            
            env.policy.update()
            if nextState[0] >= len(dfCs):
                    break
            dfr.at[nextState[0],'POA'] = reward
            dfr.at[state[0],'angle'] = state[1]
            state = nextState

            if nextState[0]%1000 == 999:
                print(path, nextState[0], env.policy.epsilon, 'samples:', len(dfr[variables_to_store].dropna()))
                dfr[variables_to_store].dropna().to_csv(
                    path + f'checkpoint_{index}.csv'
                    )
    except:
        pass
    dfr[variables_to_store].to_csv(
        path + f'checkpoint_{index}.csv'
        )