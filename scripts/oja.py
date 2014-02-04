import nengo
import build

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from nengo.matplotlib import rasterplot
from mytools import hrr, nf, fh, nengo_plot_helper, extract_probe_data
import random
import pickle

def simulate(fname, params):
    return _simulate(fname, **params.__dict__)

def _simulate(fname, seed, dim, DperE, NperD , oja_scale, oja_learning_rate, pre_tau, post_tau,
                pes_learning_rate, cleanup_params, ensemble_params, cleanup_n,
                testing_time, training_time, encoder_similarity):

    random.seed(seed)
    hrr_num = 1
    num_ensembles = int(dim / DperE)
    dim = num_ensembles * DperE
    NperE = NperD * DperE
    ttms = testing_time * 1000 #in ms

    print "Building..."
    model = nengo.Model("Network Array OJA", seed=seed)

    training_vector = np.array(hrr.HRR(dim).v)
    ortho = nf.ortho_vector(training_vector)
    ortho2 = nf.ortho_vector(training_vector)

    hrr_noise = nf.make_hrr_noise(dim, hrr_num)
    noisy_vector = hrr_noise(training_vector)
    print "HRR sim: ", np.dot(noisy_vector, training_vector)

    # --- Build input for phase 1
    gens1 = [
            nf.interpolator(1, ortho, training_vector,
               lambda x: np.true_divide(x, ttms)),
            nf.interpolator(1, training_vector,
               ortho, lambda x: np.true_divide(x, ttms)),
            nf.interpolator(1, ortho, noisy_vector,
               lambda x: np.true_divide(x, ttms)),
            nf.interpolator(1, noisy_vector,
               ortho, lambda x: np.true_divide(x, ttms)),
            nf.interpolator(1, ortho2, training_vector,
               lambda x: np.true_divide(x, ttms)),
            nf.interpolator(1, training_vector,
               ortho2, lambda x: np.true_divide(x, ttms)),
            ]
    times1 = [testing_time] * 6
    phase1_input = nf.make_f(gens1, times1)

    # --- Build input for phase 2
    gens2 = [
            nf.output(100, True, training_vector, False),
            nf.output(100, True, np.zeros(dim), False),
            nf.interpolator(1, ortho, training_vector,
               lambda x: np.true_divide(x, ttms)),
            nf.interpolator(1, training_vector,
               ortho, lambda x: np.true_divide(x, ttms)),
            nf.interpolator(1, ortho, noisy_vector,
               lambda x: np.true_divide(x, ttms)),
            nf.interpolator(1, noisy_vector,
               ortho, lambda x: np.true_divide(x, ttms)),
            nf.interpolator(1, ortho2, training_vector,
               lambda x: np.true_divide(x, ttms)),
            nf.interpolator(1, training_vector,
               ortho2, lambda x: np.true_divide(x, ttms)),
            ]

    times2 = [0.9 * training_time , 0.1 * training_time] + [testing_time] * 6
    phase2_input = nf.make_f(gens2, times2)

    orthos = [ortho]
    if cleanup_n > 1:
        orthos.append(ortho2)
    if cleanup_n > 2:
        orthos.extend([nf.ortho_vector(training_vector) for i in range(max(cleanup_n - 2, 0))])

    p = encoder_similarity
    encoders = [p * training_vector + (1-p) * o for o in orthos]
    encoders = np.array([enc / np.linalg.norm(enc) for enc in encoders])
    print [np.dot(enc,training_vector) for enc in encoders]


    # PHASE 1 - ******************************************
    print "Building phase 1"

    # ----- Make Nodes -----
    model = nengo.Model("Phase 1", seed=seed)

    inn = nengo.Node(output=phase1_input)

    cleanup = nengo.Ensemble(label='cleanup', neurons=nengo.LIF(cleanup_n), dimensions=dim,
                          encoders=encoders, **cleanup_params)

    pre_ensembles, pre_decoders, pre_connections = \
            build.build_cleanup_oja(model, inn, cleanup, DperE, NperD, num_ensembles,
                                    ensemble_params, oja_learning_rate, oja_scale,
                                    use_oja=False)
    cleanup1_s = nengo.Probe(cleanup, 'spikes')
    inn_p = nengo.Probe(inn, 'output')

    # ----- Run and get data-----
    print "Running phase 1"
    sim1 = nengo.Simulator(model, dt=0.001)
    sim1.run(sum(times1))



    # PHASE 2 - *******************************************
    print "Building phase 2"

    # ----- Make Nodes -----
    model = nengo.Model("Phase 2", seed=seed)
    with model:
        inn = nengo.Node(output=phase2_input)

        cleanup = nengo.Ensemble(label='cleanup', neurons=nengo.LIF(cleanup_n), dimensions=dim,
                              encoders=encoders, **cleanup_params)

        pre_ensembles, pre_decoders, pre_connections = \
                build.build_cleanup_oja(model, inn, cleanup, DperE, NperD, num_ensembles,
                                        ensemble_params, oja_learning_rate, oja_scale,
                                        pre_decoders=pre_decoders, pre_tau=pre_tau, post_tau=post_tau,
                                        end_time=training_time,)

        cleanup2_s = nengo.Probe(cleanup, 'spikes')

        if 0:
            weight_probes = [nengo.Probe(pc, 'transform') for pc in pre_connections]

    # ----- Run and get data-----
    print "Running phase 2"

    sim2 = nengo.Simulator(model, dt=0.001)
    sim2.run(sum(times2))

    t1 = sim1.trange()
    t2 = sim2.trange()

    sim_func = lambda x: np.dot(x, training_vector)
    sims, sims_time = extract_probe_data(t1, sim1, inn_p, func=sim_func)
    before_spikes, before_time = extract_probe_data(t1, sim1, cleanup1_s, spikes=True)

    test_slice = np.index_exp[-len(t1):][0]
    after_spikes, after_time = extract_probe_data(t2, sim2, cleanup2_s, slice=test_slice, spikes=True)

    data = dict(sims = sims, sims_time = sims_time, before_spikes = before_spikes, before_time = before_time, after_spikes = after_spikes, after_time = after_time)

    with open(fname, 'wb') as f:
        pickle.dump(data, f)

def plot(data_fname, plot_fname, params, show=False):

    with open(data_fname, 'rb') as f:
        data = pickle.load(f)

    _plot(data_fname, plot_fname, params, show=show, **data)

def _plot(data_fname, plot_fname, params, sims, sims_time, before_spikes,
            before_time, after_spikes, after_time, show=False):
    print "Plotting..."

    fig = plt.figure()
    plt.subplots_adjust(hspace=0.05)
    #plt.rc('text', usetex=True)
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rc('font', size=14)

    xticks = [2 * i * params.testing_time for i in range(4)]

    time = before_time
    spacing = 4
    title_fontsize = 20

    ax = plt.subplot(211)
    ax_spike = ax.twinx()
    ax.plot(sims_time, sims, color="Gray")
    spikes = np.concatenate((np.zeros((time.size, spacing)), before_spikes), axis=1)
    rasterplot(time, spikes, ax=ax_spike, color="Black")

    ax.set_ylim((0.0, 1.05))
    ax.set_ylabel(r'$Similarity$', fontsize=title_fontsize)
    ax.set_yticks([0,1])
    ax.set_xticks(xticks)
    ax.xaxis.set_ticklabels([])
    ax_spike.set_yticks([])

    ax = plt.subplot(212)
    ax_spike = ax.twinx()
    ax.plot(sims_time, sims, color="Gray")
    spikes = np.concatenate((np.zeros((time.size, spacing)), after_spikes), axis=1)
    rasterplot(time, spikes, ax=ax_spike, color="Black")

    ax.xaxis.set_tick_params(pad=8)
    ax.set_xlabel(r'$Time\ (s)$', fontsize=title_fontsize)
    ax.set_xticks(xticks)
    ax.set_ylabel(r'$Similarity$', fontsize=title_fontsize)
    ax.set_ylim((0.0, 1.05))
    ax.set_yticks([0,1])
    ax_spike.set_yticks([])

    plt.savefig(plot_fname)

    if show:
        plt.show()

