# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 23:17:14 2025

@author: User
"""

import yaml
import argparse
from nnets import Model_Implementation


def main():
    parser = argparse.ArgumentParser(description="Learning start")
    parser.add_argument('--par_dir',
                        default=r'./parameters/parameters_infer.yaml',
                        type=str,
                        help='path to the parameter yaml file')
     
    args = parser.parse_args()
     
    with open(args.par_dir, 'r') as f:
        par_dict = yaml.load(f, Loader=yaml.SafeLoader)
         
    model = Model_Implementation(path_data_model=par_dict['path_data_model'],
                                 mode=par_dict['mode'],
                                 output_size=par_dict['output_size'],
                                 device=par_dict['device'])
    if par_dict['mode'] == 'test':
        model.predict(model.test_data[par_dict['data_num']])
    elif par_dict['mode'] == 'predict':
        model.predict( par_dict['data'])


if __name__ == "__main__":
    
    main()