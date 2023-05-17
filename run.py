import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys

#from tangle.analysis import TangleAnalysator

from sklearn.model_selection import ParameterGrid

#############################################################################
############################# Parameter section #############################
#############################################################################

# a 的影响 
# perence 的效果

params = {
    # model setting
    'dataset': ['cifar10'],   # is expected to be one value to construct default experiment name
    'datadir': ['./data/cifar10'],
    'model': ['cnn'],      # is expected to be one value to construct default experiment name
    'lr': [0.01],
    'num_users': [100],
    'frac': [0.1],
    'num_classes': [10],
    'local_bs': [10],
    'local_ep': [11],
    'local_rep_ep': [1],
    # run setting
    'num_rounds': [100],
    'eval_every': [5],
    'src_tangle_dir': [''],         # Set to '' to not use --src-tangle-dir parameter
    'start_round': [0],
    'target_accuracy': [1],
    # node setting
    'num_tips': [2],
    'sample_size': [2],
    'publish_if_better_than': ['PARENTS'], # or PARENTS
    'reference_avg_top': [1],
    # tip selection
    'tip_selector': ['lazy_accuracy'],
    'acc_tip_selection_strategy': ['WALK'],
    'acc_cumulate_ratings': ['False'],
    'acc_ratings_to_weights': ['ALPHA'],
    'acc_select_from_weights': ['WEIGHTED_CHOICE'],
    'acc_alpha': [10],
    'use_particles': ['False'],
    'particles_depth_start': [10],
    'particles_depth_end': [20],
    'particles_number': [10],
    #poison setting
    'poison_type': ['disabled'],
    'poison_fraction': [0],
    'poison_from': [0],
    'poison_num': [2],
    
}

##############################################################################
########################## End of Parameter section ##########################
##############################################################################

def main():
    setup_filename = '1_setup.log'
    console_output_filename = '2_training.log'

    # exit_if_repo_not_clean()

    args = parse_args()
    experiment_folder = prepare_exp_folder(args)

    print("[Info]: Experiment results and log data will be stored at %s" % experiment_folder)

    run_and_document_experiments(args, experiment_folder, setup_filename, console_output_filename)

def exit_if_repo_not_clean():
    proc = subprocess.Popen(['git', 'status', '--porcelain'], stdout=subprocess.PIPE)

    try:
        dirty_files, errs = proc.communicate(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        _, errs = proc.communicate()
        print('[Error]: Could not check git status!: %s' % errs, file=sys.stderr)
        exit(1)

    if dirty_files:
        print('[Error]: You have uncommited changes. Please commit them before continuing. No experiments will be executed.', file=sys.stderr)
        exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description='Run and document an experiment.')
    parser.add_argument('--name', help='The name of the experiment. Results will be stored under ./experiments/<name>. Default: <dataset>-<model>-<exp_number>')
    parser.add_argument('--overwrite_okay', type=bool, default=False, help='Overwrite existing experiment with same name. Default: False')
    args = parser.parse_args()

    return args

def prepare_exp_folder(args):
    experiments_base = './experiments'
    os.makedirs(experiments_base, exist_ok=True)

    if not args.name:
        default_prefix = "%s-%s" % (params['dataset'][0], params['model'][0])

        # Find other experiments with default names
        all_experiments = next(os.walk(experiments_base))[1]
        default_exps = [exp for exp in all_experiments if re.match("^(%s-\d+)$" % default_prefix, exp)]

        # Find the last experiments with default name and increment id
        if len(default_exps) == 0:
            next_default_exp_id = 0
        else:
            default_exp_ids = [int(exp.split("-")[-1]) for exp in default_exps]
            default_exp_ids.sort()
            next_default_exp_id = default_exp_ids[-1] + 1

        args.name = "%s-%d" % (default_prefix, next_default_exp_id)

    exp_name = args.name

    experiment_folder = experiments_base + '/' + exp_name

     # check, if existing experiment exists
    if (os.path.exists(experiment_folder) and not args.overwrite_okay):
        print('[Error]: Experiment "%s" already exists! To overwrite set --overwrite_okay to True' % exp_name, file=sys.stderr)
        exit(1)

    os.makedirs(experiment_folder, exist_ok=True)

    return experiment_folder


def run_and_document_experiments(args, experiments_dir, setup_filename, console_output_filename):

    shutil.copy(__file__, experiments_dir)

    parameter_grid = ParameterGrid(params)
    print(f'Starting experiments for {len(parameter_grid)} parameter combinations...')
    for idx, p in enumerate(parameter_grid):
        # Create folder for that run
        experiment_folder = experiments_dir + '/config_%s' % idx
        os.makedirs(experiment_folder, exist_ok=True)

        # Prepare execution command
        command = 'python -m tangle.lab ' \
            '-dataset %s ' \
            '-datadir %s ' \
            '-model %s ' \
            '-lr %s ' \
            '--num_users %s ' \
            '--frac %s ' \
            '--num_classes %s ' \
            '--local_bs %s ' \
            '--local_ep %s ' \
            '--local_rep_ep %s ' \
            '--num-rounds %s ' \
            '--eval-every %s ' \
            '--start-from-round %s ' \
            '--target-accuracy %s ' \
            '--num-tips %s ' \
            '--sample-size %s ' \
            '--publish-if-better-than %s ' \
            '--reference-avg-top %s ' \
            '--tip-selector %s ' \
            '--acc-tip-selection-strategy %s ' \
            '--acc-cumulate-ratings %s ' \
            '--acc-ratings-to-weights %s ' \
            '--acc-select-from-weights %s ' \
            '--acc-alpha %s ' \
            '--use-particles %s ' \
            '--particles-depth-start %s ' \
            '--particles-depth-end %s ' \
            '--particles-number %s ' \
            '--poison-type %s ' \
            '--poison-fraction %s ' \
            '--poison-from %s ' \
            '--tangle-dir %s ' \
            '--poison_num %s ' 
        parameters = (
            p['dataset'],
            p['datadir'],
            p['model'],
            p['lr'],
            p['num_users'],
            p['frac'],
            p['num_classes'],
            p['local_bs'],
            p['local_ep'],
            p['local_rep_ep'],
            p['num_rounds'],
            p['eval_every'],
            p['start_round'],
            p['target_accuracy'],
            p['num_tips'],
            p['sample_size'],
            p['publish_if_better_than'],
            p['reference_avg_top'],
            p['tip_selector'],
            p['acc_tip_selection_strategy'],
            p['acc_cumulate_ratings'],
            p['acc_ratings_to_weights'],
            p['acc_select_from_weights'],
            p['acc_alpha'],
            p['use_particles'],
            p['particles_depth_start'],
            p['particles_depth_end'],
            p['particles_number'],
            p['poison_type'],
            p['poison_fraction'],
            p['poison_from'],
            experiment_folder + '/tangle_data',
            p['poison_num']
        )
        command = command.strip() % parameters

        if len(p['src_tangle_dir']) > 0:
            command = '%s --src-tangle-dir %s' % (command, p['src_tangle_dir'])

        start_time = datetime.datetime.now()

        # Print Parameters and command
        with open(experiment_folder + '/' + setup_filename, 'w+') as file:
            print('', file=file)
            print('StartTime: %s' % start_time, file=file)
            print('Parameters:', file=file)
            print(json.dumps(p, indent=4), file=file)
            print('Command: %s' % command, file=file)

        # Execute training
        print('Training started...')
        with open(experiment_folder + '/' + console_output_filename, 'w+') as file:
            
            start = p['start_round']
            end = p['num_rounds']
            print(f"Running {start} to {end}...")
            training = subprocess.Popen(command, stdout=file, stderr=file)
            training.wait()
            if training.returncode != 0:
                raise Exception('Training subprocess failed')

                

        # Document end of training
        print('Training finished. Documenting results...')
        with open(experiment_folder + '/' + setup_filename, 'a+') as file:
            end_time = datetime.datetime.now()
            print('EndTime: %s' % end_time, file=file)
            print('Duration Training: %s' % (end_time - start_time), file=file)
        '''
        print('Analysing tangle...')
        os.makedirs(experiment_folder + '/tangle_analysis', exist_ok=True)
        analysator = TangleAnalysator(experiment_folder + '/tangle_data', p['num_rounds'] - 1, experiment_folder + '/tangle_analysis')
        analysator.save_statistics(include_reference_statistics=(params['publish_if_better_than'] is 'REFERENCE'))
        '''
if __name__ == "__main__":
    main()
