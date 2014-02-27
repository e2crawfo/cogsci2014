
from association_network import LearnableAssociationNetwork, Parameters
from vectorized_graph import VectorizedGraph
import logging
import pickle

def learn(data_fname, model_fname, params, simple=False):

    cleanup_n = params.neurons_per_vector * params.num_vectors
    params.cleanup_n = cleanup_n

    lan = LearnableAssociationNetwork()
    lan.set_parameters(params)

    vg = VectorizedGraph(params.dim, params.num_vectors, params.seed, simple, save=True, draw=True)
    lan.set_vectorized_graph(vg)

    lan.build()
    sim_length, address_func, stored_func = vg.training_schedule(params.training_time)
    lan.learn(sim_length, address_func, stored_func)

    lan.save_learned_data(model_fname)

    data = lan.extract_data()

    with open(data_fname, 'wb') as f:
        pickle.dump(data, f)

