#dodo.py

from doit.cmd_base import ModuleTaskLoader
from doit.doit_cmd import DoitMain

import numpy as np
from scripts import learn, test, plot, association_network

dim = 32
seed = 100
num_vectors = [10, 20, 30, 40, 50]
training_time = 1.0
testing_time = 0.5
num_tests = 10
path_length = 3


params = association_network.Parameters(
                seed=seed,
                dim=dim,
                neurons_per_vector = 20,
                DperE = 32,
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

learn_results = ['results/learn_results_D_%g_N_%g' % (dim, N) for N in num_vectors]

#def task_oja_plot():
#    return {
#            'actions':[],
#            'filedep':[],
#            'targets':[],
#           }
#
#def task_learn_cleanup():
#    return
#
#def task_test_cleanup():
#    return
#
#def task_plot_cleanup():
#    return
#
def task_learn():
    for lr, N in zip(learn_results, num_vectors):
        yield  {
                'name':'learn_D_%g_N_%g' % (dim, N),
                'actions':[(learn.learn, [lr, params, N, training_time])],
                'file_dep':[],
                'targets':[lr],
               }

def task_edge_tests():
    for er, lr, N in zip(edge_results, learn_results, num_vectors):
        yield  {
                'name':'edge_results_D_%g_N_%g' % (dim, N),
                'actions':[(test.test_edges, [lr, er, testing_time, num_tests])],
                'file_dep':[lr],
                'targets':[er],
               }

def task_edge_accuracy_plot():
    filenames = edge_results

    return {
            'actions':[(plot.edge_accuracy_plot, [filenames])],
            'file_dep':filenames,
            'targets':['plots/edge_accuracy_plot.png'],
           }


def task_edge_similarity_plot():
    filenames = edge_results

    return {
            'actions':[(plot.edge_similarity_plot, [filenames])],
            'file_dep':filenames,
            'targets':['plots/edge_similarity_plot.png'],
           }

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
