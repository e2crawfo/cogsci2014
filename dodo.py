#dodo.py

from doit.tools import run_once

import numpy as np
from scripts import learn, test, plot, association_network

DOIT_CONFIG = {'verbosity': 2}

dim = 8
DperE = 8
seed = 100
num_vectors = [1,2,3,4,5]
training_time = 1.0
testing_time = 0.5
num_tests = 3
path_length = 2

params = association_network.Parameters(
                seed=seed,
                dim=dim,
                DperE=DperE,
                neurons_per_vector = 20,
                NperD = 30,
                oja_scale = np.true_divide(2,1),
                oja_learning_rate = np.true_divide(1,50),
                pre_tau = 0.03,
                post_tau = 0.03,
                pes_learning_rate = np.true_divide(1,1),
                cleanup_params = {'radius':1.0,
                                   'max_rates':[400],
                                   'intercepts':[0.1]},
                ensemble_params = {'radius':1.0,
                                   'max_rates':[400],
                                   'intercepts':[0.1]},
                )

edge_results = ['results/edge_test_results_D_%g_N_%g' % (dim, N) for N in num_vectors]
path_results = ['results/path_test_results_D_%g_N_%g_L_%g' % (dim, N, path_length) for N in num_vectors]

learned_networks = ['learned_networks/learn_results_D_%g_N_%g' % (dim, N) for N in num_vectors]
learn_results = ['results/learn_results_D_%g_N_%g' % (dim, N) for N in num_vectors]

edge_simulation_plots = ['plots/edge_simulation_plot_D_%g_N_%g.pdf' % (dim, N) for N in num_vectors]

def task_learn():
    for ln, lr, N in zip(learned_networks, learn_results, num_vectors):
        yield  {
                'name':'learn_D_%g_N_%g' % (dim, N),
                'actions':[(learn.learn, [lr, ln, params, N, training_time])],
                'file_dep':[],
                'targets':[lr, ln],
                'uptodate':[run_once]
               }

def task_edge_tests():
    for er, ln, N in zip(edge_results, learned_networks, num_vectors):
        yield  {
                'name':'edge_results_D_%g_N_%g' % (dim, N),
                'actions':[(test.test_edges, [ln, er, testing_time, num_tests])],
                'file_dep':[ln],
                'targets':[er]
               }

def task_edge_accuracy_plot():
    return {
            'actions':[(plot.edge_accuracy_plot, [edge_results])],
            'file_dep':edge_results,
            'targets':['plots/edge_accuracy_plot.png']
           }


def task_edge_similarity_plot():
    return {
            'actions':[(plot.edge_similarity_plot, [edge_results])],
            'file_dep':edge_results,
            'targets':['plots/edge_similarity_plot.png']
           }

# plots of the learning and testing, so we can see what happened
def task_edge_simulation_plot():
    for sp, er, lr, N in zip(edge_simulation_plots, edge_results, learn_results, num_vectors):
        yield  {
                'name':'edge_simulation_plot_D_%g_N_%g' % (dim, N),
                'actions':[(plot.simulation_plot, [sp, lr, er])],
                'file_dep':[er, lr],
                'targets':[sp]
               }

def task_example_simulation_plot():
    N = 5
    num_tests = N

    dim = 32
    DperE = 32
    seed = 100
    training_time = 1.0
    testing_time = 0.5

    params = association_network.Parameters(
                    seed=seed,
                    dim=dim,
                    DperE=DperE,
                    neurons_per_vector = 20,
                    NperD = 30,
                    oja_scale = np.true_divide(2,1),
                    oja_learning_rate = np.true_divide(1,50),
                    pre_tau = 0.03,
                    post_tau = 0.03,
                    pes_learning_rate = np.true_divide(1,1),
                    cleanup_params = {'radius':1.0,
                                       'max_rates':[400],
                                       'intercepts':[0.1]},
                    ensemble_params = {'radius':1.0,
                                       'max_rates':[400],
                                       'intercepts':[0.1]},
                    )

    learn_fname = 'example_learn_D_%g_N_%g' % (dim, N)
    learn_data = 'results/' + learn_fname
    learn_network = 'learned_networks/' + learn_fname

    test_fname = 'results/example_test_D_%g_N_%g' % (dim, N)
    plot_fname = 'plots/example_plot_D_%g_N_%g.pdf' % (dim, N)

    yield  {
            'name':'example_learn_D_%g_N_%g' % (dim, N),
            'actions':[(learn.learn, [learn_data, learn_network, params, N, training_time, True])],
            'file_dep':[],
            'targets':[learn_data, learn_network],
            'uptodate':[run_once]
           }

    test_order = [N-1] + range(N-1)
    yield  {
            'name':'example_test_D_%g_N_%g' % (dim, N),
            'actions':[(test.test_edges, [learn_network, test_fname, testing_time, num_tests, test_order])],
            'file_dep':[learn_network],
            'targets':[test_fname]
           }

    yield  {
            'name':'example_simulation_plot_D_%g_N_%g' % (dim, N),
            'actions':[(plot.simulation_plot, [plot_fname, learn_data, test_fname])],
            'file_dep':[test_fname, learn_data],
            'targets':[plot_fname]
           }

#def task_oja_plot():
#    return {
#            'actions':[],
#            'filedep':[],
#            'targets':[],
#           }

    #def task_path_tests():
    #    for pr, lr, N in zip(path_results, learn_results, num_vectors):
    #        yield  {
    #                'name':'path_results_D_%g_N_%g_L_%g' % (dim, N, path_length),
    #                'actions':[test.test_path(lr, pr, testing_time, num_tests, path_length)],
    #                'filedep':[lr],
    #                'targets':[er],
    #               }
    #
    #def task_path_accuracy_plot():
    #    filenames = path_results
    #
    #    return {
    #            'actions':[plot.path_accuracy_plot(filenames)],
    #            'filedep':filenames,
    #            'targets':['plots/path_accuracy_plot.png'],
    #           }
    #
    #def task_path_similarity_plot():
    #    filenames = path_results
    #    return {
    #            'actions':[plot.path_similarity_plot(filenames)],
    #            'filedep':filenames,
    #            'targets':['plots/path_similarity_plot.png'],
    #           }
    #
