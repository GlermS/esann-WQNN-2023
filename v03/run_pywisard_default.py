import utils.run_tool as run
from pywisard.binarization.thermometer import *
from pywisard.models.regression import RegWisard
import os
from utils.files import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from multiprocessing import Pool


model_type = 'ReW-pywisard'
search_mode = 'decay'
with open('setup.json','r') as file:
    setup = json.loads(file.read())[model_type]


inputs = 'ghi-zenith-azimuth'
filename = f'results/{search_mode}_{model_type}_{inputs}.json'


dfCs, encoders = run.init(*setup['period'])

def encode_func(ctx):
    a = list(encoders['ghi'].encode(ctx['mghi'])) + list(encoders['zenith'].encode(ctx['zenith']))+ list(encoders['azimuth'].encode(ctx['azimuth'])) + list(encoders['angle'].encode(ctx['angle']))
    return np.array(a)




def exec(config):
    s_params, decay_rate, learning_rate, forget , tuple_size= config
    input_size = len(encode_func({'mghi': 0, 'zenith':0, 'azimuth':0, 'angle':0}))

    path = f'results/{inputs}/{search_mode}/{model_type}/k={s_params["k"]}_decay-rate={decay_rate}_learning-rate={learning_rate}_forget-factor={forget}_tuple-size={tuple_size}/'
    os.makedirs(path, exist_ok=True)

    def ModelGenerator():
        model = RegWisard(input_size, tuple_size, forget_factor = forget)
        return model
    
    run.run_individual(dfCs, ModelGenerator, encode_func, path, searchMode={'mode':search_mode, 'params':s_params}, decay_rate = decay_rate, learning_rate=decay_rate)

if __name__ == '__main__':
    configs = []

    for learning_rate in setup['params']['learning_rate']:
        for decay_rate in setup['params']['decay_rate']:
            for tuple_size in setup['params']['tuple_size']:
                for forget_factor in setup['params']['forget_factor']:
                    for s_params in setup['search'][search_mode]:
                        configs.append((s_params, decay_rate, learning_rate, forget_factor, tuple_size))


    with Pool(2) as p:
        print(p.map(exec, configs[::-1]))