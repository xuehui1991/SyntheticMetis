import argparse
import json
import math
import numpy as np 
import os
import sys
import logging
import pandas as pd
import timeit

sys.path.append('./')

from metis_tuner.metis_tuner import MetisTuner

# set the logger format
log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    filename="systhetic-metis.log",
    filemode="a",
    level=logging.DEBUG,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)

LOG = logging.getLogger('sythetic-metis')

def load_json(in_path):
    with open(in_path) as file:
        return json.load(file)


def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df


def get_euclidean_distance(param, row):
    result = 0
    for key in ['x', 'y', 'z']:
        try:
            result = result + math.pow((param[key] - row[key]), 2)
        except:
            LOG.debug(param[key])
            LOG.debug(row[key])
            raise
    return math.sqrt(result)


def get_similar_point(df, param):
    min_distance = sys.maxsize
    min_row = None

    x = param['x']
    y = param['y']
    z = param['z']

    for _, row in df.iterrows():
        distance = get_euclidean_distance(param, row)
        if min_distance > distance:
            min_distance = distance
            min_row = row

    if min_row is None:
        raise RuntimeError("min_row cannot be None.")
    
    # print("min_row")
    # print(min_row)
    return min_row['reward_rescale']


def get_fake_result(params):
    x = params['x']
    y = params['y']
    # make an assumption that 'z' is the fiexed value  
    qps = params['z']
    # the envoriment

    noise = np.random.normal(0, 0.01, 3)
    loss = math.pow(x - 3 + noise[0], 2) * math.pow(y -4 + noise[1], 2) * math.pow(qps - 5 + noise[2], 2) /2.0
    return loss


def run(args, tuner: str = 'Metis', max_trial_num : int = 1000, step_size : int = 10, csv_path : str = 'simulation_data.csv'):
    ''' example code for Metis Tuner
    
    Arguments:
        args -- argparse
    
    Keyword Arguments:
        tuner {str} -- [tuner name] (default: {'Metis'})
        max_trial_num {int} -- [max numbers of trial] (default: {1000})
        step_size {int} -- [step size to use fixed_param] (default: {10})
    
    Raises:
        NotImplementedError -- The Tuner has not been implemented yet
    '''

    # load search space
    search_space = load_json(args.search_space)

    # init tuner
    if tuner == 'Metis':
        tuner = MetisTuner(optimize_mode="minimize", no_candidates=False)
    else:
        raise NotImplementedError("Tuner {} is not implemented yet".format(tuner))

    tuner.update_search_space(search_space)
    df = load_csv(csv_path)

    trial_num = 0
    for trial_num in range(1,max_trial_num+1):
        LOG.info("trial_num:" + str(trial_num))

        if trial_num % 50 != 0:        
            # ask tuner a parameter
            params_dict = tuner.generate_parameters(topK=1)
            best_params, next_params_topK = params_dict["best_config"], params_dict["next_config_topK"]

            LOG.info("best_params:")
            LOG.info(best_params)
            LOG.info("next_params_topk:")
            LOG.info("### next: " + str(next_params_topK[0]))

            RECEIVED_PARAMS = next_params_topK[0]

            cost = get_similar_point(df, next_params_topK[0])
            LOG.info("Take the next_params and get result:")
            LOG.info("### cost: " + str(cost))

            tuner.receive_trial_result(RECEIVED_PARAMS, cost)
        else:
            # contexts = {'platform': ['Mac'-0, 'Windows'-1], 
            #             'network': ['wifi'-0, 'wired'-1], 
            #             'country': ['US'-0, 'CA'-1]}
            unique_contexts = [{'platform':0, 'network':0, 'country':0}, 
                               {'platform':1, 'network':0, 'country':0},
                               {'platform':0, 'network':1, 'country':0},
                               {'platform':1, 'network':1, 'country':0},
                               {'platform':0, 'network':0, 'country':1}, 
                               {'platform':1, 'network':0, 'country':1},
                               {'platform':0, 'network':1, 'country':1},
                               {'platform':1, 'network':1, 'country':1},]

            for context in unique_contexts:
                LOG.info("context:")
                LOG.info(context)

                fix_param = []
                for key in context:
                    fix_param.append((key, context[key]))

                params_dict = tuner.generate_parameters(topK=1, fixed_params= fix_param)
                best_params, next_params_topK = params_dict["best_config"], params_dict["next_config_topK"]
                
                LOG.info("best_params:")
                LOG.info("### best: " + str(best_params))
                LOG.info("next_params_topk:")
                LOG.info(next_params_topK)
                RECEIVED_PARAMS = best_params

                cost = get_similar_point(df, RECEIVED_PARAMS)
                LOG.info("Take the best params and get result:")
                LOG.info("### cost: " + str(cost))

                tuner.receive_trial_result(RECEIVED_PARAMS, cost)
            
    LOG.info("Train Done.")


def test():
    csv_path = 'simulation_data.csv'
    df = load_csv(csv_path)

    start = timeit.default_timer()
    param = {'x':3.419200 , 'y':1.671143 , 'z':1.242625 }
    print(get_similar_point(df, param))

    stop = timeit.default_timer()
    print('Time: ', stop - start)  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--search_space', type=str, default='./synthetic_searchspc.json', help="Path to search space file")
    parser.add_argument('--tuner', default='Metis', help="Tuner algorithms to use")

    args = parser.parse_args()

    if args.search_space is not None:
        assert os.path.exists(args.search_space)
    
    run(args)

    #test()
