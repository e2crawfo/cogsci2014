#test a created network
import pickle

from association_network import LearnableAssociationNetwork, Parameters
from vectorized_graph import VectorizedGraph

import logging

def test_edges(in_fname, out_fname, testing_time, num_tests, order=None):
    logging.basicConfig(filename='log/'+out_fname.split('/')[-1]+'.log')

    lan = LearnableAssociationNetwork()
    lan.load_learned_data(in_fname)

    vg = lan.vectorized_graph

    schedule = vg.edge_testing_schedule(testing_time, num_tests, order)
    sim_length, address_func, correct_vectors, input_vectors = schedule

    lan.test(sim_length, address_func)

    data = lan.extract_data()
    data['input_vectors'] = input_vectors
    data['correct_vectors'] = correct_vectors
    data['testing_time'] = testing_time
    data['num_tests'] = num_tests
    data['num_vectors'] = vg.G.number_of_nodes()

    with open(out_fname, 'wb') as f:
        pickle.dump(data, f)

def test_paths(in_fname, out_fname, testing_time, num_tests, path_length):
    logging.basicConfig(filename='log/'+out_fname.split('/')[-1]+'.log')

    lan = LearnableAssociationNetwork()
    lan.load_learned_data(in_fname)

    vg = lan.vectorized_graph

    schedule = vg.path_testing_schedule(testing_time, num_tests, path_length)
    sim_length, address_func, correct_vectors = schedule

    lan.test(sim_length, address_func)

    data = lan.extract_data()
    data['correct_vectors'] = correct_vectors
    data['testing_time'] = testing_time
    data['num_tests'] = num_tests
    data['num_vectors'] = vg.G.number_of_nodes()

    with open(out_fname, 'wb') as f:
        pickle.dump(data, f)

