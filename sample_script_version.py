from datetime import datetime
import pandas as pd
import argparse
import os
import json

# these are just dummy functions that don't really do anything
class Model():
    def __init__(self, name, output_dir):
        self.name = name
        self.output_dir = output_dir
        
    def train(self, data):
        print('trained :)')
        
    def save(self):
        with open(self.output_dir + self.name, 'w') as f:
            f.write('pickled_model')
            
    def save_results(self):
        with open(self.output_dir + self.name + ' results', 'w') as f:
            f.write('results')
    
    def output(self, data):
        return 0.25
            
def complicated_preprocessing_1(df):
    return df

def complicated_preprocessing_2(df):
    return df


def save_experiment_params(FLAGS):
    flags_as_dict = vars(FLAGS)
    json_filename = os.path.join(FLAGS.output_dir, "config.json")
    with open(json_filename, 'w') as fp:
        json.dump(flags_as_dict, fp, indent=4)

def interact(outcome_var_name, interact_var_1, interact_var_2, df):
    df[outcome_var_name] = df[interact_var_1]*df[interact_var_2]
    return df

preprocessing_methods = {
    -1: lambda x: x,
    1: complicated_preprocessing_1,
    2: complicated_preprocessing_2,
}

def main(FLAGS):
    # create log directory if it doesn't exist
    os.makedirs(FLAGS.output_dir)

    # save the experiment params
    save_experiment_params(FLAGS)


    df = pd.read_csv(FLAGS.dataset_path)

    assert ('fuel_tank_capacity' in list(df.columns)), 'data does not contain required feature'

    if FLAGS.add_interaction == True:
        df = interact(FLAGS.interaction_name, FLAGS.interact_var1, FLAGS.interact_var2, df)

    for method in FLAGS.preprocessing_methods:
        df = preprocessing_methods[method](df)

    
    model = Model(FLAGS.model_type, FLAGS.output_dir)
    model.train(df)
    model.save()
    assert (model.output(df) < 1 & model.output(df) > 0), 'model output is in the wrong range'
    model.save_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_path',
                        type=str,
                        help='Path to the dataset you want to use')

    parser.add_argument('--add_interaction',
                        type=bool,
                        default=False,
                        help='If you want to add an interaction')


    parser.add_argument('--interaction_name',
                        type=str,
                        default='',
                        help='Name of interaction variable you want to add')
    parser.add_argument('--interact_var1',
                        type=str,
                        default='',
                        help='Name of 1st variable you want to interact')
    parser.add_argument('--interact_var2',
                        type=str,
                        default='',
                        help='Name of 2nd variable you want to interact')


    parser.add_argument('--preprocessing_methods',
                        type=int,
                        nargs='+',
                        default=-1,
                        help='what transformations/etc to apply to the data. By default, does nothing.\
                             Can chain multiple steps together.')

    parser.add_argument('--model_type',
                        type=str,
                        default='LinearRegression',
                        help='type of model you want to use')

    parser.add_argument('--output_dir',
                        type=str,
                        default='./output_dir/{}_{}/'.format(datetime.now().strftime('%m%d'), datetime.now().strftime('%H%M')),
                        help='name of output directory')
    
    parser.add_argument('--description',
                        type=str,
                        default='',
                        help='description of your experiment')
    
    FLAGS = parser.parse_args()
    main(FLAGS)
                