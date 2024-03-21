import pickle
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def run_hyperopt_loop(hyperparameter_optimiser, space, trials_file_name, num_runs_before_saving, num_loops):
    #coded that runs hyperopt and saves results as trials, to allow for easy reloading also
    #first checks for already existing trial file, load and set increased max evals
    try:
        with open(trials_file_name, 'rb') as f:
            trials = pickle.load(f)
        print(f'Loaded previous {len(trials.trials)} trials results')
    #if not trials exists, make a new object
    except:
        trials = Trials()
    #loop through the number of times requested
    for i in range(num_loops):
        #set number of evaluations for within fmin to be the number specified before a save takes place
        max_evals = len(trials.trials) + num_runs_before_saving
        #run fmin as usual
        best = fmin(fn= hyperparameter_optimiser, space=space, algo=tpe.suggest, max_evals=max_evals, trials = trials)
        #saves trials object as a checkpoint to be reloaded if required to save starting opt again from scratch
        with open(trials_file_name, 'wb') as f:
            pickle.dump(trials, f)