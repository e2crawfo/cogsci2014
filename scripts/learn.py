
from association_network import LearnableAssociationNetwork, Parameters
from vectorized_graph import VectorizedGraph

def learn():
    seed = 81223
    training_time = 1 #in seconds
    testing_time = 0.5
    dim = 32

    neurons_per_vector = 20
    num_vectors = 5
    cleanup_n = neurons_per_vector * num_vectors

    #cleanup_intercept = 0.14
    cleanup_intercept = cutilities.minimum_threshold(0.95, neurons_per_vector/2, cleanup_n, dim)
    ensemble_intercept = 0.1
params = Parameters( seed=seed,
        dim=dim,
        DperE = 32,
        NperD = 30,
        oja_scale = np.true_divide(2,1),
        oja_learning_rate = np.true_divide(1,50),
        pre_tau = 0.03,
        post_tau = 0.03,
        pes_learning_rate = np.true_divide(1,1),
        cleanup_params = {'radius':1.0,
                           'max_rates':[400],
                           'intercepts':[cleanup_intercept]},
        ensemble_params = {'radius':1.0,
                           'max_rates':[400],
                           'intercepts':[ensemble_intercept]},
        eint = ensemble_intercept,
        cint = ensemble_intercept,
        )

    lan = LearnableAssociationNetwork()
    lan.set_vectorized_graph(params)

    vg = VectorizedGraph(dim, num_vectors, seed)
    lan.set_vectorized_graph(vg)

    lan.build()
    sim_length, address_func, stored_func = vg.training_schedule(training_time)
    lan.learn(sim_length, address_func, stored_func)

    data_title = 'learned_weights'
    data_filename = params.make_filename(data_title, dir=data_title)

    lan.save_learned_data(data_filename)

if __name__ == "__main__":
    learn()

