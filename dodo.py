#dodo.py

from doit.tools import run_once

import numpy as np
import copy
from scripts import learn, test, plot, association_network, oja

DOIT_CONFIG = {'verbosity': 2}

dim = 32
DperE = 32
num_samples = 5
seed = 100
num_vectors = range(2,10)
training_time = 1.0
testing_time = 0.5
num_tests = 5
path_length = 2

params = association_network.Parameters(
                dim=dim,
                seed=seed,
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

run_configs = [(dim, N, s) for N in num_vectors for s in range(num_samples)]
edge_results = ['results/edge_test_results_D_%g_N_%g_s_%g' % rc for rc in run_configs]

learned_networks = ['learned_networks/learn_results_D_%g_N_%g_s_%g' % rc for rc in run_configs]
learn_results = ['results/learn_results_D_%g_N_%g_s_%g' % rc for rc in run_configs]

edge_simulation_plots = ['plots/edge_simulation_plot_D_%g_N_%g_s_%g.pdf' % rc for rc in run_configs]

def task_learn():
    for ln, lr, rc in zip(learned_networks, learn_results, run_configs):
        cur_params = copy.deepcopy(params)
        cur_params.seed = params.seed + rc[2]
        yield  {
                'name':ln,
                'actions':[(learn.learn, [lr, ln, cur_params, N, training_time])],
                'file_dep':[],
                'targets':[lr, ln],
                'uptodate':[run_once]
               }

def task_edge_tests():
    for er, ln in zip(edge_results, learned_networks):
        yield  {
                'name':er,
                'actions':[(test.test_edges, [ln, er, testing_time, num_tests])],
                'file_dep':[ln],
                'targets':[er]
               }

# plots of the learning and testing, so we can see what happened
def task_edge_simulation_plot():
    for sp, er, lr in zip(edge_simulation_plots, edge_results, learn_results):
        yield  {
                'name':sp,
                'actions':[(plot.simulation_plot, [sp, lr, er])],
                'file_dep':[er, lr],
                'targets':[sp]
               }

def task_edge_accuracy_plot():
    edge_acc_plot = 'plots/edge_acc_plot_D_%g.pdf' % (dim)
    return {
            'actions':[(plot.edge_accuracy_plot, [edge_results, run_configs, edge_acc_plot])],
            'file_dep':edge_results,
            'targets':[edge_acc_plot]
           }

def task_edge_similarity_plot():
    edge_sim_plot = 'plots/edge_sim_plot_D_%g.pdf' % (dim)
    return {
            'actions':[(plot.edge_similarity_plot, [edge_results, run_configs, edge_sim_plot])],
            'file_dep':edge_results,
            'targets':[edge_sim_plot]
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
            'actions':[(learn.learn, [learn_data, learn_network, params, N, training_time, seed, True])],
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


def task_oja():
    dim = 64
    DperE = 64
    seed = 510

    params = association_network.Parameters(
                    seed=seed,
                    dim=dim,
                    DperE=DperE,
                    NperD = 30,
                    oja_scale = np.true_divide(7,1),
                    oja_learning_rate = np.true_divide(1,50),
                    pre_tau = 0.03,
                    post_tau = 0.03,
                    pes_learning_rate = np.true_divide(1,1),
                    cleanup_params = {'radius':1.0,
                                       'max_rates':[200],
                                       'intercepts':[0.2]},
                    ensemble_params = {'radius':1.0,
                                       'max_rates':[400],
                                       'intercepts':[0.1]},
                    cleanup_n = 1,
                    testing_time = 0.1,
                    training_time = 2,
                    encoder_similarity=0.3
                    )

    data_fname = "results/oja_results_D_%g" % dim
    plot_fname = "plots/oja_plot_D_%g.pdf" % dim

    yield {
            'name':'sim_oja_D_%g' % dim,
            'actions':[(oja.simulate, [data_fname, params])],
            'file_dep':[],
            'targets':[data_fname],
            'uptodate':[run_once]
           }


    yield {
            'name':'plot_oja_D_%g' % dim,
            'actions':[(oja.plot, [data_fname, plot_fname, params])],
            'file_dep':[data_fname],
            'targets':[plot_fname],
           }

#path_results = ['results/path_test_results_D_%g_N_%g_L_%g' % (dim, N, path_length) for N in num_vectors]
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
