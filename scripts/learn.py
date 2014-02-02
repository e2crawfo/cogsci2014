
from association_network import LearnableAssociationNetwork, Parameters
from vectorized_graph import VectorizedGraph
import cutilities
import logging
import pickle

def learn(data_fname, model_fname, params, num_vectors, training_time, log=True):

    logging.basicConfig(filename='log/'+data_fname.split('/')[-1]+'.log')

    cleanup_n = params.neurons_per_vector * num_vectors
    params.cleanup_n = cleanup_n
    prob, cleanup_intercept = \
            cutilities.minimum_threshold(0.5, params.neurons_per_vector/2, cleanup_n, params.dim)
    params.cleanup_params['intercepts'] = [cleanup_intercept]

    lan = LearnableAssociationNetwork()
    lan.set_parameters(params)

    vg = VectorizedGraph(params.dim, num_vectors, params.seed)
    lan.set_vectorized_graph(vg)

    lan.build()
    sim_length, address_func, stored_func = vg.training_schedule(training_time)
    lan.learn(sim_length, address_func, stored_func)

    lan.save_learned_data(model_fname)

    data = lan.extract_data()
    data['vg'] = vg
    with open(data_fname, 'wb') as f:
        pickle.dump(data, f)

