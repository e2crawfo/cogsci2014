
from association_network import LearnableAssociationNetwork, Parameters
from vectorized_graph import VectorizedGraph
import cutilities

def learn(fname, params, num_vectors, training_time):

    cleanup_n = params.neurons_per_vector * num_vectors
    params.cleanup_n = cleanup_n
    prob, cleanup_intercept = \
            cutilities.minimum_threshold(0.95, params.neurons_per_vector/2, cleanup_n, params.dim)
    params.cleanup_params['intercepts'] = cleanup_intercept

    lan = LearnableAssociationNetwork()
    lan.set_parameters(params)

    vg = VectorizedGraph(params.dim, num_vectors, params.seed)
    lan.set_vectorized_graph(vg)

    lan.build()
    sim_length, address_func, stored_func = vg.training_schedule(training_time)
    lan.learn(sim_length, address_func, stored_func)

    #data_title = 'learned_weights'
    #data_filename = params.make_filename(data_title, dir=data_title)

    lan.save_learned_data(fname)

