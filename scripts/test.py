#test a created network
import pickle

from association_network import LearnableAssociationNetwork, Parameters
from vectorized_graph import VectorizedGraph

def test_edges(fname, testing_time, num_tests):
    p = Parameters()

    lan = LearnableAssociationNetwork()
    lan.set_parameters(p)

    lan.build()
    lan.load_learned_data(fname)

    vg = lan.vectorized_graph

    schedule = vg.edge_testing_schedule(testing_time, num_tests)
    sim_length, address_func, stored_func = schedule

    lan.test(sim_length, address_func, stored_func)

    data = lan.extract_data()

    with open(fname, 'rb'):
        pickle.dump(data)

def test_paths(fname, testing_time, num_tests, path_length):
    p = Parameters()

    lan = LearnableAssociationNetwork()
    lan.set_parameters(p)

    lan.build()
    lan.load_learned_data(fname)

    vg = lan.vectorized_graph

    schedule = vg.path_testing_schedule(testing_time, num_tests, path_length)
    sim_length, address_func, stored_func = schedule

    lan.test(sim_length, address_func, stored_func)

    data = lan.extract_data()

    with open(fname, 'rb'):
        pickle.dump(data)

