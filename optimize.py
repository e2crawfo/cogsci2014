import time
from hyperopt import fmin, tpe, hp, Trials
from hyperopt.mongoexp import MongoTrials
import subprocess
import argparse
import os
import sys
import numpy as np
from scipy import stats
from mytools import fh
from scripts import association_network

def make_objective(base_params, num_samples, num_vectors, num_tests, training_time, testing_time):
    import copy
    from mytools import fh
    from scripts import learn, test, analyze
    from hyperopt import STATUS_OK
    import numpy as np
    from scipy import stats
    def objective(kwargs):

        params = copy.deepcopy(base_params)
        params.oja_scale = kwargs['oja_scale']
        params.oja_learning_rate = kwargs['oja_learning_rate']
        params.ensemble_params['intercepts'] = kwargs['ens_intercept']
        params.cleanup_params['intercepts'] = kwargs['cleanup_intercept']
        params.cleanup_params['radius'] = kwargs['radius']
        params.tau_rc = kwargs['tau_rc']
        params.tau_ref = kwargs['tau_ref']
        params.ctau_rc = kwargs['ctau_rc']
        params.ctau_ref = kwargs['ctau_ref']
        params.seed = kwargs['seed']

        mean_input_sims = []
        mean_output_sims = []

        for i in range(num_samples):
            data_fname = fh.make_filename('training', config_dict=params.__dict__, use_time=True)
            model_fname = fh.make_filename('models', config_dict=params.__dict__, use_time=True)
            test_fname = fh.make_filename('tests', config_dict=params.__dict__, use_time=True)

            learn.learn(data_fname, model_fname, params, num_vectors, training_time, simple='simple2')
            test.test_edges(model_fname, test_fname, testing_time, num_tests)

            analysis = analyze.analyze_edge_test_similarity(test_fname)
            input_sims, output_sims, data = analysis

            mean_input_sims.append(np.mean(input_sims))
            mean_output_sims.append(np.mean(output_sims))

        return {
            'loss': -np.mean(mean_output_sims),
            'loss_variance': stats.sem(mean_output_sims),
            'status': STATUS_OK,
            }

    return objective

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimize parameters for learning an associative memory.')

    parser.add_argument('--mongo-workers',
            dest='mongo_workers',
            default=8,
            type=int,
            help='Number of parallel workers to use to evaluate points.'\
                  ' Only has an effect if --mongo is also supplied')
    parser.add_argument('--exp-key',
            dest='exp_key',
            default='exp1',
            type=str,
            help='Unique key identifying this experiment within the mongodb')
    parser.add_argument('--dry-run',
            dest='dry_run',
            default=False,
            action='store_true',
            help='A dry run will not evaluate the function. Useful for testing the'\
            'hyperopt framework without having to wait for the function evaluation')

    argvals = parser.parse_args()

    dim = 8
    DperE = 8

    num_samples = 5
    num_vectors = 10
    num_tests = 10
    training_time = 1.0
    testing_time = 0.5

    params = association_network.Parameters(
                    dim=dim,
                    DperE=DperE,
                    neurons_per_vector = 20,
                    NperD = 30,
                    pre_tau = 0.03,
                    post_tau = 0.03,
                    pes_learning_rate = np.true_divide(1,1),
                    cleanup_params = {'radius':1.0,
                                       'max_rates':[400],
                                       'intercepts':[0.1]},
                    ensemble_params = {'radius':np.sqrt(np.true_divide(DperE, dim)),
                                       'max_rates':[400],
                                       'intercepts':[0.1]},
                    )

    num_mongo_workers = max(argvals.mongo_workers, 1)

    exp_key = argvals.exp_key
    dry_run = argvals.dry_run

    if dry_run:
        def make_f():
            from hyperopt import STATUS_OK
            def f(x):
                return { 'loss': 0, 'loss_variance': 0, 'status': STATUS_OK}
            return f
        objective = make_f()
    else:
        objective = make_objective(params, num_samples, num_vectors, num_tests,
                                    training_time, testing_time)

    trials = MongoTrials('mongo://localhost:1234/assoc/jobs', exp_key=exp_key)
    print "Trials: " + str(trials.trials) + "\n"
    print "Results: " + str(trials.results) + "\n"
    print "Losses: " + str(trials.losses()) + "\n"
    print "Statuses: " + str(trials.statuses()) + "\n"

    worker_call_string = \
        ["hyperopt-mongo-worker",
        "--mongo=localhost:1234/assoc",
        "--max-consecutive-failures","2",
        "--reserve-timeout", "2.0",
        #"--workdir","~/cogsci2014/",
        ]

    print "Worker Call String"
    print worker_call_string
    workers = []
    for i in range(num_mongo_workers):
        p = subprocess.Popen(worker_call_string)
        workers.append(p)

    space = {
            'seed':hp.randint('seed', 1000000),
            'oja_scale':hp.uniform('oja_scale', 1, 5),
            'radius':hp.uniform('radius', .65, .75),
            'oja_learning_rate':hp.uniform('oja_learning_rate', 0.01, 0.05),
            'ens_intercept':hp.uniform('ens_intercept', 0.08, 0.15),
            'cleanup_intercept':hp.uniform('cleanup_intercept', 0.09, 0.16),
            'tau_rc':hp.uniform('tau_rc', 0.08, 0.14),
            'tau_ref':hp.uniform('tau_ref', 0.003, 0.005),
            'ctau_rc':hp.uniform('ctau_rc', 0.25, 0.4),
            'ctau_ref':hp.uniform('ctau_ref', 0.002, 0.005),
            }

    then = time.time()

    print "Calling fMin"
    best = fmin(objective,
        space=space,
        algo=tpe.suggest,
        max_evals=20,
        trials=trials)
    print "Done fMin"
    print "Best: ", best

    now = time.time()

    directory = '/data/e2crawfo/cleanuplearning/opt/logs'
    filename = fh.make_filename('optlog', directory=directory, use_time=True)
    aggregated_log = open(filename, 'w')

    aggregated_log.write("Time for fmin: " + str(now - then) + "\n")
    aggregated_log.write("Trials: " + str(trials.trials) + "\n")
    aggregated_log.write("Results: " + str(trials.results) + "\n")
    aggregated_log.write("Losses: " + str(trials.losses()) + "\n")
    aggregated_log.write("Statuses: " + str(trials.statuses()) + "\n")

    aggregated_log.close()


    for p in workers:
       p.terminate()

