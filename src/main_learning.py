import yaml
import argparse
from nnets import Model_Implementation


def main():
    parser = argparse.ArgumentParser(description="Learning start")
    parser.add_argument('--par_dir', 
                        default=r'./parameters/parameters_learning.yaml',
                        type=str, 
                        help='path to the parameter yaml file')
     
    args = parser.parse_args()
     
    with open(args.par_dir, 'r') as f:
        par_dict = yaml.load(f, Loader=yaml.SafeLoader)
         
    model = Model_Implementation(path_data_model=par_dict['path_data_model'],
                                 mode=par_dict['mode'],
                                 batch_size=par_dict['batch_size'],
                                 output_size=par_dict['output_size'],
                                 device=par_dict['device'])
    model.start_fit(par_dict['EPOCHS'])


if __name__ == "__main__":
    
    main()