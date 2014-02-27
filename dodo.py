#dodo.py

from doit.tools import run_once

import numpy as np
import copy
from scripts import learn, test, plot, association_network, oja
from mytools import fh

DOIT_CONFIG = {'verbosity': 2}

dr = '/data/e2crawfo/cleanuplearning/run/'

num_vectors = range(4,7)
num_tests = 10
testing_time = 0.5
num_samples = 1

params = association_network.Parameters(
                seed=100,
                #dim=128,
                #DperE=32,
                dim=8,
                DperE=8,
                neurons_per_vector = 30,
                NperD = 30,
                oja_scale = 15.30946097,
                oja_learning_rate = 0.01043522,
                pre_tau = 0.03,
                post_tau = 0.03,
                training_time=1.0,
                pes_learning_rate = np.true_divide(1,1),
                cleanup_params = {'radius':0.62345,
                                   'max_rates':[400],
                                   'intercepts':[0.32056287]},
                )

params.ensemble_params = {'radius':np.sqrt(np.true_divide(params.DperE, params.dim)),
                          'max_rates':[400],
                          'intercepts':[0.2278348]}

dim = params.dim

run_configs = [(params.dim, N, s) for N in num_vectors for s in range(num_samples)]
learn_results = [fh.make_filename('learn_D_%g_N_%g_s_%g' % rc, dr+'learn', use_time=False) for rc in run_configs]
learned_networks = [fh.make_filename('network_D_%g_N_%g_s_%g' % rc, dr+'networks', use_time=False) for rc in run_configs]
edge_test_results = [fh.make_filename('edge_test_D_%g_N_%g_s_%g' % rc, dr+'tests', use_time=False) for rc in run_configs]
edge_test_plots = [fh.make_filename('edge_test_plots_D_%g_N_%g_s_%g' % rc, dr+'plots', use_time=False) for rc in run_configs]

edge_test_plots = [dr + 'plots/edge_simulation_plot_D_%g_N_%g_s_%g.pdf' % rc for rc in run_configs]

def task_learn():
    for ln, lr, rc in zip(learned_networks, learn_results, run_configs):
        cur_params = copy.deepcopy(params)
        cur_params.num_vectors = rc[1]
        cur_params.seed = params.seed + rc[2]

        yield  {
                'name':ln,
                'actions':[(learn.learn, [lr, ln, cur_params, 'simple2'])],
                'file_dep':[],
                'targets':[lr, ln],
                'uptodate':[run_once]
               }

def task_edge_tests():
    for er, ln in zip(edge_test_results, learned_networks):
        yield  {
                'name':er,
                'actions':[(test.test_edges, [ln, er, testing_time, num_tests])],
                'file_dep':[ln],
                'targets':[er]
               }

def task_edge_simulation_plot():
    for sp, er, lr in zip(edge_test_plots, edge_test_results, learn_results):
        yield  {
                'name':sp,
                'actions':[(plot.simulation_plot, [sp, lr, er])],
                'file_dep':[er, lr],
                'targets':[sp]
               }

def task_edge_accuracy_plot():
    edge_acc_plot = 'plots/edge_acc_plot_D_%g.pdf' % (dim)
    return {
            'actions':[(plot.edge_accuracy_plot, [edge_test_results, run_configs, edge_acc_plot])],
            'file_dep':edge_test_results,
            'targets':[edge_acc_plot]
           }

def task_edge_similarity_plot():
    edge_sim_plot = 'plots/edge_sim_plot_D_%g.pdf' % (dim)
    return {
            'actions':[(plot.edge_similarity_plot, [edge_test_results, run_configs, edge_sim_plot])],
            'file_dep':edge_test_results,
            'targets':[edge_sim_plot]
           }

def task_example_simulation_plot():

    num_tests = 5
    testing_time = 0.5

    params = association_network.Parameters(
                    seed=257938,
                    dim=8,
                    DperE = 8,
                    #dim=64,
                    #DperE = 32,
                    num_vectors = 5,
                    neurons_per_vector = 30,
                    NperD = 50,
                    pre_tau = 0.03,
                    post_tau = 0.03,
                    training_time = 1.0,
                    tau_ref=0.0048327,
                    tau_rc=0.09689,
                    ctau_ref=0.00257,
                    ctau_rc=0.27103,
                    oja_scale=20.221052987,
                    oja_learning_rate=0.045654,
                    pes_learning_rate = np.true_divide(1,1),
                    cleanup_params = {'radius':0.716534,
                                       'max_rates':[200],
                                       'intercepts':[0.133256]},
                    )

    params.ensemble_params = {'radius':np.sqrt(np.true_divide(params.DperE, params.dim)),
                              'max_rates':[200],
                              'intercepts':[0.098351]}

    dim = params.dim
    num_vectors = params.num_vectors

    learn_data_fname = fh.make_filename('example_learn', directory=dr+'learn', config_dict=params.__dict__, use_time=False)
    learn_model_fname = fh.make_filename('example_network', directory=dr+'networks', config_dict=params.__dict__, use_time=False)
    test_fname = fh.make_filename('example_tests', directory=dr+'tests', config_dict=params.__dict__, use_time=False)
    plot_fname = fh.make_filename('example_plots', directory=dr+'plots', config_dict=params.__dict__, use_time=False)
    plot_fname += ".pdf"

    yield  {
            'name':'example_learn_D_%g_num_vectors_%g' % (dim, num_vectors),
            'actions':[(learn.learn, [learn_data_fname, learn_model_fname, params, 'simple2'])],
            'file_dep':[],
            'targets':[learn_data_fname, learn_model_fname],
            'uptodate':[run_once]
           }

    yield  {
            'name':'example_test_D_%g_num_vectors_%g' % (dim, num_vectors),
            'actions':[(test.test_edges, [learn_model_fname, test_fname, testing_time, num_tests])],
            'file_dep':[learn_model_fname],
            'targets':[test_fname]
           }

    yield  {
            'name':'example_simulation_plot_D_%g_num_vectors_%g' % (dim, num_vectors),
            'actions':[(plot.simulation_plot, [plot_fname, learn_data_fname, test_fname])],
            'file_dep':[test_fname, learn_data_fname],
            'targets':[plot_fname]
           }


def task_oja():
    params = association_network.Parameters(
                    seed=510,
                    dim=8,
                    DperE=8,
                    #dim=64,
                    #DperE=64,
                    NperD = 30,
                    oja_scale = np.true_divide(7,1),
                    oja_learning_rate = np.true_divide(1,50),
                    pre_tau = 0.03,
                    post_tau = 0.03,
                    pes_learning_rate = np.true_divide(1,1),
                    cleanup_n = 1,
                    testing_time = 0.1,
                    training_time = 2,
                    encoder_similarity=0.3,
                    cleanup_params = {'radius':1.0,
                                       'max_rates':[200],
                                       'intercepts':[0.2]},
                    ensemble_params = {'radius':1.0,
                                       'max_rates':[400],
                                       'intercepts':[0.1]},
                    )

    dim = params.dim

    data_fname = dr + "tests/oja_results_D_%g" % dim
    plot_fname = dr + "plots/oja_plot_D_%g.pdf" % dim

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

#path_length = 2
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

#paper parameters
#    N = 5
#    num_tests = N
#
#    dim = 32
#    DperE = 32
#    seed = 100
#    training_time = 1.0
#    testing_time = 0.5
#
#    params = association_network.Parameters(
#                    dim=dim,
#                    seed=seed,
#                    DperE=DperE,
#                    neurons_per_vector = 20,
#                    NperD = 30,
#                    oja_scale = np.true_divide(2,1),
#                    oja_learning_rate = np.true_divide(1,50),
#                    pre_tau = 0.03,
#                    post_tau = 0.03,
#                    pes_learning_rate = np.true_divide(1,1),
#                    cleanup_params = {'radius':1.0,
#                                       'max_rates':[400],
#                                       'intercepts':[0.1]},
#                    ensemble_params = {'radius':1.0,
#                                       'max_rates':[400],
#                                       'intercepts':[0.1]},
#                    )

