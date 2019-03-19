import torch
import torch.nn as nn
from filter_estimator import FilterEstimator
from pwise import Piecewise


def get_single_generator(params):
    # define generator
    gen_type = params['type']
    if gen_type == 'filter': return FilterEstimator(params['params'])
    elif gen_type == 'piecewise':  return Piecewise(params['params'])
    else: raise ValueError('Generator not available.')


def get_generator(params):
    # initialize generator
    generator = []
    # if it is a list
    if isinstance(params,list):
        # it's a concatenation of generators
        for i in range(len(params)):
            generator.append(get_single_generator(params[i]))
        # make it a Sequential
        generator = nn.Sequential(*generator)
    else:
        # it's a single generator
        generator = get_single_generator(params)
    # return generator
    return generator


def get_single_generator_name(params):
    # define output filename
    gp = params['params']
    pieces = [params['type']]
    for key in sorted(gp.keys()): pieces.append(key + '_' + str(gp[key]))
    return '_'.join(pieces)


def get_generator_name(params):
    # initialize generator
    generator_name = ''
    # if it is a list
    if isinstance(params,list):
        # it's a concatenation of generators
        for i in range(len(params)):
            if i>0: generator_name += '__'
            generator_name += get_single_generator_name(params[i])
    else:
        # it's a single generator
        generator_name = get_single_generator_name(params)
    # return generator
    return generator_name
