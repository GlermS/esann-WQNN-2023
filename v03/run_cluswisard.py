from utils.files import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import utils.run_tool as run
import wisardpkg as wsd
import json
from multiprocessing import Pool

model_type = 'CReW'
search_mode = 'decay'
with open('setup.json','r') as file:
    setup = json.loads(file.read())[model_type]


inputs = 'ghi-zenith-azimuth'
filename = f'results/{search_mode}_{model_type}_{inputs}.json'


dfCs, encoders = run.init(*setup['period'])


def encode_func(ctx):
    a = list(encoders['ghi'].encode(ctx['mghi'])) + list(encoders['zenith'].encode(ctx['zenith']))+ list(encoders['azimuth'].encode(ctx['azimuth'])) + list(encoders['angle'].encode(ctx['angle']))
    return wsd.BinInput(a)



def exec(config):
    s_params, decay_rate, learning_rate, mean_type, tuple_size = config
    path = f'results/{inputs}/{search_mode}/{model_type}/k={s_params["k"]}_decay-rate={decay_rate}_learning-rate={learning_rate}_type={mean_type}_tuple-size={tuple_size}/'
    os.makedirs(path, exist_ok=True)

    def ModelGenerator():
        model = wsd.ClusRegressionWisard('''{
        "addressSize": ''' + str(tuple_size) + ''',
        "minOnes": 3,
        "mean": {
            "type": "''' + mean_type + '''"
        }
        }''')
        a = list(encoders['ghi'].encode(0)) + list(encoders['zenith'].encode(0))+ list(encoders['azimuth'].encode(0)) + list(encoders['angle'].encode(0))
        model.train(wsd.BinInput(a), 0)
        return model
        
    run.run_individual(dfCs, ModelGenerator, encode_func, path, searchMode={'mode':search_mode, 'params':s_params}, decay_rate = decay_rate, learning_rate=decay_rate)

if __name__ == '__main__':
    configs = []

    for learning_rate in setup['params']['learning_rate']:
        for decay_rate in setup['params']['decay_rate']:
            for tuple_size in setup['params']['tuple_size']:
                for mean_type in setup['params']['mean_type']:
                    for s_params in setup['search'][search_mode]:
                        configs.append((s_params, decay_rate, learning_rate, mean_type, tuple_size))



    with Pool(2) as p:
        print(p.map(exec, configs))