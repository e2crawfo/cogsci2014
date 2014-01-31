#test a created network
import pickle

from association_network import LearnableAssociationNetwork, Parameters
from vectorized_graph import VectorizedGraph

def test_edges(in_fname, out_fname, testing_time, num_tests):
    lan = LearnableAssociationNetwork()
    lan.load_learned_data(in_fname)

    vg = lan.vectorized_graph

    schedule = vg.edge_testing_schedule(testing_time, num_tests)
    sim_length, address_func, correct_vectors = schedule

    lan.test(sim_length, address_func)

    data = lan.extract_data()
    data['correct_vectors'] = correct_vectors

    with open(out_fname, 'rb'):
        pickle.dump(data)

def test_paths(in_fname, out_fname, testing_time, num_tests, path_length):
    lan = LearnableAssociationNetwork()
    lan.load_learned_data(in_fname)

    vg = lan.vectorized_graph

    schedule = vg.path_testing_schedule(testing_time, num_tests, path_length)
    sim_length, address_func, correct_vectors = schedule

    lan.test(sim_length, address_func)

    data = lan.extract_data()
    data['correct_vectors'] = correct_vectors

    with open(out_fname, 'rb'):
        pickle.dump(data)

